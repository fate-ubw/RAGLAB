import pdb
import re
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
class SelfAsk(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
    
    def init(self, args):
        self.selfask_max_iter = args.selfask_max_iter

    def infer(self, query:str) -> tuple[str, dict]:
        '''
        Instruction: our self-ask instructions strictly follow the instructions provided by the original paper and open-source code. 
                     Considering the window length of local llm(llama2-7b ,mistral, etc.), we only used one-shot instead of using the four-shot in the self ask open source code.
        paper:[https://arxiv.org/abs/2210.03350]        
        github code: [https://github.com/ofirpress/self-ask]
        '''
        target_instruction = self.find_algorithm_instruction('self_ask-followup_question', self.task)
        input_with_followup = target_instruction.format_map({'query': query})
        output_list = self.llm.generate(input_with_followup)
        Output = output_list[0]
        follow_up = Output.text

        print(f'follow up question:{follow_up}')
        generation_track = {}
        turn_idx = 1
        if 'Follow up:' in follow_up :
            while 'Follow up:' in follow_up and turn_idx < self.selfask_max_iter: 
                turn_idx += 1
                followup_question = self._extract_followup(follow_up)
                if followup_question == '':
                    print(f'Bad case!!!')
                    break
                passages = self.retrieval.search(followup_question)
                passages = self._truncate_passages(passages)
                collated_passages = self.collate_passages(passages)
                target_instruction = self.find_algorithm_instruction('self_ask-read', self.task) # 但是这个回答的是中间的问题，
                input_with_passages = target_instruction.format_map({'passages': collated_passages, 'query': followup_question})
                output_list = self.llm.generate(input_with_passages)
                Output = output_list[0]
                intermediate_answer = Output.text
                generation_track[turn_idx] = {
                                                'follow up question': followup_question,
                                                'intermediate answer': intermediate_answer,
                                                'cite passages': passages
                                              }
                input_with_followup = input_with_followup + follow_up + ' \n Intermediate Answer: ' + intermediate_answer + ' \n '
                output_list = self.llm.generate(input_with_followup)
                Output = output_list[0]
                follow_up = Output.text
            # end of while
            if 'So the final answer is:' in follow_up: 
                follow_up = self._extract_final_answer_1(follow_up)
            elif 'Final Answer:' in follow_up:
                follow_up = self._extract_final_answer_2(follow_up)
            elif follow_up == '':
                # some special case will generate ''. In this situation we need add instruction for self ask finish the whole inference
                output_list = self.llm.generate(input_with_followup + 'So the final answer is:')
                Output = output_list[0]
                follow_up = Output.text
            else:
                print(f'reach max iteration!!!')
                output_list = self.llm.generate(input_with_followup + 'So the final answer is:')
                Output = output_list[0]
                follow_up = Output.text
        else:
            passages = self.retrieval.search(query)
            passages = self._truncate_passages(passages)
            collated_passages = self.collate_passages(passages)
            target_instruction = self.find_algorithm_instruction('self_ask-read', self.task)
            input = target_instruction.format_map({'passages': collated_passages, 'query': query})
            output_list = self.llm.generate(input)
            Output = output_list[0]
            follow_up = Output.text

            generation_track['cite passages'] = passages
        generation_track['final answer'] = follow_up # 
        if self.task == 'PubHealth':
            final_answer = self._postprocess_pubhealth(follow_up)
        elif self.task == 'Feverous':
            final_answer = self._postprocess_feverous(follow_up)
        elif self.task == 'StrategyQA':
            final_answer = self._postprocess_strategyQA(follow_up)
        else:
            final_answer = follow_up
        return final_answer, generation_track

    def _postprocess_strategyQA(self, answer:str)->str:
        wrong_pattern_1 = "Are follow up questions needed here"
        if wrong_pattern_1 in answer:
            real_answer, _, _ = answer.partition(wrong_pattern_1)
            if  'No' in real_answer:
                processed_answer = 'False'
            elif 'Yes' in real_answer:
                processed_answer = 'True'
            else:
                processed_answer = real_answer
        elif 'Yes' in answer:
            processed_answer = 'True'                
        elif 'No' in answer:
            processed_answer = 'False'
        else:
            processed_answer = answer
        return processed_answer

    def _postprocess_pubhealth(self, answer:str)->str:
        wrong_pattern_1 = "Are follow up questions needed here"
        if wrong_pattern_1 in answer:
            real_answer, _, _ = answer.partition(wrong_pattern_1)
            if  'No' in real_answer:
                processed_answer = 'false'
            elif 'Yes' in real_answer:
                processed_answer = 'true'
            else:
                processed_answer = real_answer
        elif 'Yes' in answer:
            processed_answer = 'true'                
        elif 'No' in answer:
            processed_answer = 'false'
        else:
            processed_answer = answer
        return processed_answer

    def _postprocess_feverous(self, answer:str)->str:
        wrong_pattern_1 = "Are follow up questions needed here"
        if wrong_pattern_1 in answer:
            real_answer, _, _ = answer.partition(wrong_pattern_1)
            if  'No' in real_answer:
                processed_answer = 'REFUTES'
            elif 'Yes' in real_answer:
                processed_answer = 'SUPPORTS'
            else:
                processed_answer = real_answer
        elif 'Yes' in answer:
            processed_answer = 'SUPPORTS'
        elif 'No' in answer:
            processed_answer = 'REFUTES'
        else:
            processed_answer = answer
        return processed_answer

    def _extract_followup(self, followup):
        followup_pattern = r'Follow up: (.+)'
        result = re.findall(followup_pattern, followup)
        followup_question = ''
        if len(result) >= 1:
            followup_question = result[0]
        return followup_question

    def _extract_final_answer_1(self, followup):
        followup_pattern = r'So the final answer is: (.+)'
        result = re.findall(followup_pattern, followup, re.DOTALL) # re.DOTALL flag instructs the regex engine to allow . to match any character, including newline \n.
        followup_question = ''
        if len(result) >= 1:
            followup_question = result[0]
        return followup_question
    
    def _extract_final_answer_2(self, followup):
        followup_pattern = r'Final Answer: (.+)'
        result = re.findall(followup_pattern, followup, re.DOTALL) # re.DOTALL flag instructs the regex engine to allow . to match any character, including newline \n.
        followup_question = ''
        if len(result) >= 1:
            followup_question = result[0]
        return followup_question