import re
from typing import List, Dict, Any
from itertools import combinations
from raglab.data_collector.parallel_base_data_collector import DatasetCollectorParallel
from raglab.data_collector.base_data_collector import DatasetCollector
from raglab.language_model import OpenaiModel, HF_Model, HF_VLLM, Lora_Model, UnifiedApiRequest
from raglab.instruction_lab import ALGORITHM_INSTRUCTIONS
from raglab.retrieval import ContrieverRrtieve, ColbertRetrieve, ColbertApi
import pdb

class CriticModelDatasetCollector(DatasetCollectorParallel):

    def __init__(self,args):
        super().__init__(args)
        self.llm = self.steup_llm(args)
        self.retrieval = self.setup_retrieval(args)
        self.log_file = f"api_statistics.txt"

    def process_item(self, item: Dict[str, Any], idx: int, format:str) -> list[Dict[str, Any]]:
        collected_data = []
        dataset_type = self.find_dataset_type(self.dataset_name)
        item = self.preprocessed_data(item, dataset_type)
        input_tokens_sum = 0 
        output_tokens_sum = 0
        if format == 'flashrag':
            task_instruction = self.find_dataset_instruction(self.dataset_name)
            source_question = item.get("question","")
            answer = item.get("golden_answers",[""])[0]
        elif format == 'stanford_alpaca':
            task_instruction = item.get("instruction","")
            source_question = item.get("input","")
            answer = item.get("output","")
        question_with_instruction = f"{task_instruction} {source_question}"  # task instruction from dataset
        # Collect [Retrieve]
        target_instruction =  self.find_algorithm_instruction('collector-selfrag-[Retrieve]', None)
        input = target_instruction.format_map({'instruction':question_with_instruction})
        outputlist = self.llm.generate(input)
        # calculate algorithm
        if outputlist == []:
            return collected_data
        input_tokens_sum += outputlist[0].prompt_tokens_num
        output_tokens_sum += outputlist[0].tokens_num

        output = outputlist[0].text
        preprocessed_output =   self._preprocess_retrieval(output)
        collected_data.append({
            "instruction": self.find_algorithm_instruction('critic-retrieval_instruction', None),
            "input": self.find_algorithm_instruction('critic-retrieval_input', None).format_map({'instruction':question_with_instruction}),
            "output": preprocessed_output,
            "task":'retrieval',
            "raw_dataset": self.dataset_name,
            "raw_dataset_idx": item.get("id", idx)
            })
        if dataset_type == "MultiChoice":
            passages = self.retrieval.search(source_question)
        else:
            passages = self.retrieval.search(question_with_instruction)
        
        for _, passage in passages.items():
            target_instruction = self.find_algorithm_instruction('collector-selfrag-[IsRel]', None)
            input = target_instruction.format_map({'instruction':question_with_instruction, 'evidence': passage['title'] + '\n' + passage['text']})
            outputlist = self.llm.generate(input)
            if outputlist == []:
                return collected_data
            # calculate algorithm
            input_tokens_sum += outputlist[0].prompt_tokens_num
            output_tokens_sum += outputlist[0].tokens_num

            output = outputlist[0].text
            collected_data.append({
                "instruction": self.find_algorithm_instruction('critic-relevance_instruction', None),
                "input": self.find_algorithm_instruction('critic-relevance_input', None).format_map({'instruction':question_with_instruction,'evidence':passage['title'] + '\n' + passage['text']}),
                "output": output,
                "task":'relevance',
                "raw_dataset": self.dataset_name,
                "raw_dataset_idx": item.get("id", idx)
            })
            # Collecte [IsSup]
            target_instruction = self.find_algorithm_instruction('collector-selfrag-[IsSup]', None)
            input = target_instruction.format_map({'instruction':question_with_instruction, 'target_output':answer, 'evidence':passage['title'] +'\n'+ passage['text']})
            outputlist = self.llm.generate(input)
            if outputlist == []:
                return collected_data
            # calculate algorithm
            input_tokens_sum += outputlist[0].prompt_tokens_num
            output_tokens_sum += outputlist[0].tokens_num


            output = outputlist[0].text
            collected_data.append({
                "instruction": self.find_algorithm_instruction('critic-ground_instruction', None),
                "input": self.find_algorithm_instruction('critic-ground_input', None).format_map({'instruction':question_with_instruction, 'output': answer, 'evidence': passage['title'] + '\n' + passage['text']}),
                "output": output,
                "task":'groudness',
                "raw_dataset": self.dataset_name,
                "raw_dataset_idx": item.get("id", idx)
            })
        # --> end of top_k passages loop
        # generate sample -> [Utility:2]
        target_instruction = self.find_algorithm_instruction('collector-selfrag-[Utility]-imporve', None)
        input = target_instruction.format_map({'instruction':question_with_instruction, 'output':answer})
        outputlist = self.llm.generate(input)
        if outputlist == []:
            return collected_data
        # calculate algorithm
        input_tokens_sum += outputlist[0].prompt_tokens_num
        output_tokens_sum += outputlist[0].tokens_num

        output = outputlist[0].text
        preprocessed_output = self._preprocess_utility(output) # golden answer give [utility:2]
        preprocessed_output = self._enforce_utility_any_to_2(preprocessed_output)
        collected_data.append({
            "instruction": self.find_algorithm_instruction('critic-utility_instruction', None),
            "input": self.find_algorithm_instruction('critic-utility_input', None).format_map({'instruction':question_with_instruction, 'output':answer}),
            "output": preprocessed_output,
            "task":'utility',
            "raw_dataset": self.dataset_name,
            "raw_dataset_idx": item.get("id", idx)
        })
        # generate incorrect sample -> [utility:1]
        target_instruction = self.find_algorithm_instruction("collector-incorrect_sample", None)
        input = target_instruction.format_map({"question":question_with_instruction, "answer": answer})
        outputlist = self.llm.generate(input)
        if outputlist == []:
            return collected_data
        # calculate algorithm
        input_tokens_sum += outputlist[0].prompt_tokens_num
        output_tokens_sum += outputlist[0].tokens_num

        incorrect_answer = outputlist[0].text # create incorrect answer based on question & golden answer
        target_instruction = self.find_algorithm_instruction('collector-selfrag-[Utility]-imporve', None)
        input = target_instruction.format_map({'instruction':question_with_instruction, 'output':incorrect_answer})
        outputlist = self.llm.generate(input)
        if outputlist == []:
            return collected_data
        # calculate algorithm
        input_tokens_sum += outputlist[0].prompt_tokens_num
        output_tokens_sum += outputlist[0].tokens_num

        output = outputlist[0].text
        preprocessed_output = self._preprocess_utility(output) # golden answer give [utility:2]
        preprocessed_output = self._enforce_utility_any_to_1(preprocessed_output)
        collected_data.append({
            "instruction": self.find_algorithm_instruction('critic-utility_instruction', None),
            "input": self.find_algorithm_instruction('critic-utility_input', None).format_map({'instruction':question_with_instruction, 'output': incorrect_answer}),
            "output": preprocessed_output,
            "task":'utility',
            "raw_dataset": self.dataset_name,
            "raw_dataset_idx": item.get("id", idx)
        })
        # pair-wise
        concated_passages = self.extract_and_concatenate(passages)
        target_instruction = self.find_algorithm_instruction('collector-Most_relevantest_passages', None)
        input = target_instruction.format_map({'instruction':source_question,'evidences':concated_passages})
        outputlist = self.llm.generate(input)
        if outputlist == []:
            return collected_data
        # calculate algorithm
        input_tokens_sum += outputlist[0].prompt_tokens_num
        output_tokens_sum += outputlist[0].tokens_num

        output = outputlist[0].text
        passage_idx = self.extract_number(output)
        if passage_idx is None:
            # find no relevant passages
            return collected_data
        elif passage_idx > len(passages): # Prevent hallucinations in the output of the LLM
            return collected_data
        else:
            pass
        
        target_passages = f"Title:{passages[passage_idx]['title']} content:{passages[passage_idx]['text']}"
        collected_data.append({
            "instruction": self.find_algorithm_instruction('critic-Infer_improvement_answer', None).format_map({'instruction':question_with_instruction ,'evidences':target_passages}),
            "input": '',
            "output": str(answer),
            "task":'improvement_answer',
            "raw_dataset": self.dataset_name,
            "raw_dataset_idx": item.get("id", idx)
        })
        # pair-wise
        target_instruction = self.find_algorithm_instruction("collector-candidate_answers",None)
        input = target_instruction.format_map({"instruction":question_with_instruction, "evidences":target_passages, "answer": answer})
        outputlist = self.llm.generate(input)
        if outputlist == []:
            return collected_data
        # calculate algorithm
        input_tokens_sum += outputlist[0].prompt_tokens_num
        output_tokens_sum += outputlist[0].tokens_num

        output = outputlist[0].text
        candidate_answers = self.extract_answers(output)
        candidate_answers.append(answer)
        # get pair-wise answers
        paired_answers = self.generate_unique_pairs(candidate_answers)
        for pair in paired_answers:
            target_instruction = self.find_algorithm_instruction("collector-pair_wise", None)
            input = target_instruction.format_map({"instruction":question_with_instruction, "response_1":pair[0], "response_2":pair[1]})
            outputlist = self.llm.generate(input)
            if outputlist == []:
                return collected_data
            # calculate algorithm
            input_tokens_sum += outputlist[0].prompt_tokens_num
            output_tokens_sum += outputlist[0].tokens_num

            output_1 = outputlist[0].text
            eval_result_first_turn = self.extract_evaluation_info(output_1)
            # change position and re-evaluate
            input = target_instruction.format_map({"instruction":question_with_instruction, "response_1":pair[1], "response_2":pair[0]})
            outputlist = self.llm.generate(input)
            if outputlist == []:
                return collected_data
            # calculate algorithm
            input_tokens_sum += outputlist[0].prompt_tokens_num
            output_tokens_sum += outputlist[0].tokens_num

            output_2 = outputlist[0].text
            eval_result_second_turn = self.extract_evaluation_info(output_2)
            if eval_result_first_turn['Eval_result'] == -1 or eval_result_second_turn["Eval_result"] == -1:
                # wrong format
                continue
            elif eval_result_first_turn['Eval_result'] == 0 and  eval_result_second_turn['Eval_result'] == 0:
                # tie
                # first turn data collect
                collected_data.append({
                    "instruction": self.find_algorithm_instruction('critic-pair_wise-instruction', None).format_map({"instruction": question_with_instruction, "response_1": pair[0], "response_2": pair[1]}),
                    "input": '',
                    "output": self.find_algorithm_instruction('critic-pair_wise-output', None).format_map({"Eval_result":'tie', "explanation": eval_result_first_turn['explanation'], "Reference": eval_result_first_turn['Reference']}),
                    "task":'pair_wise',
                    "raw_dataset": self.dataset_name,
                    "raw_dataset_idx": item.get("id", idx)
                })
                # second turn data collect
                collected_data.append({
                    "instruction": self.find_algorithm_instruction('critic-pair_wise-instruction', None).format_map({"instruction":question_with_instruction, "response_1": pair[1], "response_2":pair[0]}),
                    "input": '',
                    "output": self.find_algorithm_instruction('critic-pair_wise-output', None).format_map({"Eval_result": 'tie', "explanation": eval_result_second_turn['explanation'], "Reference": eval_result_second_turn['Reference']}),
                    "task":'pair_wise',
                    "raw_dataset": self.dataset_name,
                    "raw_dataset_idx": item.get("id", idx)
                })
            elif eval_result_first_turn['Eval_result'] + eval_result_second_turn['Eval_result'] == 3:
                # consistency response
                # first turn
                collected_data.append({
                    "instruction": self.find_algorithm_instruction('critic-pair_wise-instruction', None).format_map({"instruction": question_with_instruction, "response_1": pair[0], "response_2": pair[1]}),
                    "input": '',
                    "output": self.find_algorithm_instruction('critic-pair_wise-output', None).format_map(eval_result_first_turn),
                    "task":'pair_wise',
                    "raw_dataset": self.dataset_name,
                    "raw_dataset_idx": item.get("id", idx)
                })
                # second turn
                collected_data.append({
                    "instruction": self.find_algorithm_instruction('critic-pair_wise-instruction', None).format_map({"instruction": question_with_instruction, "response_1": pair[1], "response_2": pair[0]}),
                    "input": '',
                    "output": self.find_algorithm_instruction('critic-pair_wise-output', None).format_map(eval_result_second_turn),
                    "task":'pair_wise',
                    "raw_dataset": self.dataset_name,
                    "raw_dataset_idx": item.get("id", idx)
                })
            else:
                # inconsistency response
                continue
        # # --> end of pair-wise loop
        self.log_statistics(input_tokens_sum, output_tokens_sum, collected_data)
        return collected_data

    def steup_llm(self, args):
        if  args.llm_mode == 'HF_Model':
            if  args.use_vllm:
                llm = HF_VLLM(args)
                llm.load_model() # load_model() will load local model and tokenizer  
            else:
                llm = HF_Model(args)
                llm.load_model() # load_model() will load local model and tokenizer
        elif  args.llm_mode == "Lora_Model":
            llm = Lora_Model(args)
            llm.load_model() #  load_model() will load base model and lora adapter then merged by peft to get complete model
        elif  args.llm_mode == 'Openai_api':
            llm = OpenaiModel(args)
            llm.load_model() # load_model() will load api configs and tiktoken
        elif args.llm_mode == 'Unified_api':
            llm = UnifiedApiRequest(args)
            llm.load_model()
        else:
            raise LanguageModelError("Language model must be huggingface or openai api.")
        return llm

    def setup_retrieval(self, args):
        if 'colbert' ==  args.retrieval_name:
            retrieval_model = ColbertRetrieve(args)
            retrieval_model.setup_retrieve()
        elif 'contriever' ==  args.retrieval_name:
            retrieval_model = ContrieverRrtieve(args)
            retrieval_model.setup_retrieve()
        elif  'colbert_api' ==  args.retrieval_name:
            retrieval_model = ColbertApi(args)
            retrieval_model.setup_retrieve()
        elif 'pregiven_passages' ==  args.retrieval_name:
            retrieval_model = None # no need setup retrieval model when pre-given passages prepared
        else:
            raise RetrievalModelError("invalid retrieval model")
        return retrieval_model 

    def find_algorithm_instruction(self, algorithm_name:str, dataset_name:str) -> str:
        target_instruction = ''
        for instruction in ALGORITHM_INSTRUCTIONS:
            if instruction['algorithm_name'] == algorithm_name and instruction['dataset_name'] == dataset_name:
                target_instruction = instruction['instruction']
                break
        if target_instruction == '':
            raise InstructionNotFoundError('Instruction name not recognized. Please provide a valid instruction key.')
        return target_instruction
    
    def check_dict_keys(d):
        keys = set(d.keys())
        if all(key in keys for key in ['instruction', 'input', 'output']):
            return "Executing A: Found instruction, input, and output"
        elif all(key in keys for key in ['question', 'golden_answers']):
            return "Executing B: Found question and input"
        else:
            return "No matching condition found"

    def _preprocess_retrieval(self, output: str) -> str:
        # Check for [Yes] and replace with [Retrieval]
        if re.search(r'\[Yes\]|\[YES\]|\[yes\]', output, flags=re.MULTILINE | re.IGNORECASE):
            output = re.sub(r'\[Yes\]|\[YES\]|\[yes\]', '[Retrieval] ', output, flags=re.IGNORECASE)
        
        # Check for [No] and replace with [No Retrieval]
        elif re.search(r'\[No\]|\[NO\]|\[no\]', output, flags=re.MULTILINE | re.IGNORECASE):
            output = re.sub(r'\[No\]|\[NO\]|\[no\]', '[No Retrieval] ', output, flags=re.IGNORECASE)
        
        # Handle cases where 'Yes' or 'No' is at the start of the string without brackets
        elif re.search(r'^(Yes|No)\s*(\n+)?', output, flags=re.MULTILINE | re.IGNORECASE):
            output = re.sub(r'^(Yes|No)\s*(\n+)?', 
                            lambda m: f"[{'Retrieval' if m.group(1).lower() == 'yes' else 'No Retrieval'}]{m.group(2) or ''}", 
                            output, flags=re.MULTILINE | re.IGNORECASE)

        return output


    def _preprocess_utility(self, output: str) -> str:
        # Replace "Perceived utility: X" with "[Utility:X]"
        for i in range(1, 3):
            pattern = f'Perceived utility: {i}'
            replacement = f'[Utility:{i}]'
            output = re.sub(pattern, replacement, output)
        return output
        
    def _enforce_utility_any_to_2(self, answer: str) -> str:
        if re.search(r'\[Utility:2\]$', answer, flags=re.MULTILINE):
            return answer
        else:
            # Replace the entire answer with [Utility:2]
            return '[Utility:2]'
        
    def _enforce_utility_any_to_1(self, answer: str) -> str:
        if re.search(r'\[Utility:1\]$', answer, flags=re.MULTILINE):
            return answer
        else:
            # Replace the entire answer with [Utility:1]
            return '[Utility:1]'

    def extract_and_concatenate(self, passages:dict)-> str:
        result = []
        for key, value in passages.items():
            title = value['title']
            text = value['text']
            result.append(f"[{key}]: {title} {text}")
        return "\n".join(result)

    def extract_number(self, output:str) -> int:
        # Use regular expression to find the number inside square brackets
        match = re.search(r'\[(\d+)\]', output)
        if match:
            return int(match.group(1))
        else:
            return None

    def extract_answers(self, text:str) -> str:
        # Use regular expression to find all answers labeled with [1], [2], [3], and [4]
        text = re.sub(r'\b(?:Very High Quality Answer|High Quality Answer|Low Quality Answer|Very Low Quality Answer)\b:\s*', '', text, flags=re.MULTILINE)
        matches = re.findall(r'\[\d\] ([^\[]+)', text)
        return matches

    def generate_unique_pairs(self, input_list:str)-> str:
        return [list(pair) for pair in combinations(input_list, 2)]

    def extract_evaluation_info(self, text: str) -> dict:
        # Updated regular expression to match all three formats
        pattern = r'(?:###\s*)?Evaluation:(.+?)(?:###\s*|\n)Explanation:(.+?)(?:###\s*|\n)Reference:(.+?)(?=\n###|$)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if not matches:
            return {
                'Eval_result': -1,
                'explanation': '',
                'Reference': ''
            }
        match = matches[0]
        eval_result = match[0].strip()
        explanation = match[1].strip()
        reference = match[2].strip()

        # Extract the number from "Response X" or replace with "tie" if "tie" is found
        response_match = re.search(r'(?:Response\s*)?(\d+)', eval_result, re.IGNORECASE)
        tie_match = re.search(r'\btie\b', eval_result, re.IGNORECASE)
        if response_match:
            eval_result = int(response_match.group(1))
        elif tie_match:
            eval_result = 0
        else:
            eval_result = -1
        result = {
            'Eval_result': eval_result,
            'explanation': explanation,
            'Reference': reference
        }
        return result

    def log_statistics(self, input_tokens_sum, output_tokens_sum, collected_data):
        stats = f"""
        Dataset Name: {self.dataset_name}
        Input tokens number one raw data: {input_tokens_sum}
        Output tokens number one raw data: {output_tokens_sum}
        Average input token number: {input_tokens_sum/len(collected_data)}
        Average output token number: {output_tokens_sum/len(collected_data)}
        -------------------------------------------
        """
        with open(self.log_file, 'a') as f:
            f.write(stats)


class LanguageModelError(Exception):
    pass

class RetrievalModelError(Exception):
    pass

class InstructionNotFoundError(Exception):
    pass