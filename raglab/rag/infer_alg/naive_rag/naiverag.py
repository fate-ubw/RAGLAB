import os
from tqdm import tqdm
from datetime import datetime
import time
from typing import Optional, Any
import logging
import random

from raglab.dataset.utils import get_dataset # load datasets
from raglab.retrieval import ContrieverRrtieve, ColbertRetrieve, ColbertApi
from raglab.language_model import OpenaiModel, HF_Model, HF_VLLM, Lora_Model
from raglab.instruction_lab import ALGORITHM_INSTRUCTIONS
from raglab.gpu_manager import GPUManager
import pdb
RED = '\033[91m'
END = '\033[0m'
GREEN = '\033[92m'

class NaiveRag:
    def __init__(self, args):
        self.args = args
        # gpu manager
        # if args.gpu_ids is not None: # multi gpu manager
        #     self.gpu_manager = GPUManager(args.gpu_ids)
        # output file name config
        self.config = args.config
        self.algorithm_name = args.algorithm_name
        self.llm_path = args.llm_path
        self.llm_name = args.llm_name
        # eval config
        self.task = args.task
        self.eval_datapath = args.eval_datapath
        self.output_dir = args.output_dir
        # llm config
            #(other configs check raglab/language_model/*.py files)
        self.llm_mode = args.llm_mode
        self.use_vllm = args.use_vllm
        # retrieval args
        self.n_docs = args.n_docs
        self.retrieval_name = args.retrieval_name
        self.realtime_retrieval = args.realtime_retrieval
        self.passages_max_length = args.passages_max_length
        # setup model and database 
        # if args.gpu_ids is not None:
        #     self.gpu_manager.allocate_gpu()
        self.llm = self.steup_llm(args)
        # steup retrieval
        if self.realtime_retrieval:
            self.retrieval = self.setup_retrieval(args) # retrieval model
        if self.task == '':
            self.print_fn = print
        else:
            self.print_fn = self.setup_logger(args)
        self.init(args)
        self.print_fn(f'Rag Parameter:{args}')

    def init(self, args):
        pass

    def inference(self, query: Optional[str] = None, mode = 'interact'):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            self.print_fn(f"Interactive mode: query = {query}")
            final_answer, generation_track = self.infer(query)
            return final_answer, generation_track
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self) # here we input self because dataset classed need self.print_fn
            self.eval_dataset = self.EvalData.load_dataset()
            self.print_fn(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                eval_data = self.EvalData.preprocess(eval_data) # some dataset need preprocess such as: arc_challenge
                question = eval_data[self.EvalData.InputStruction.question] 
                outputs, generation_track = self.infer(question)
                inference_results = self.EvalData.record_result(eval_data, outputs, inference_results)
                self.print_fn(f'{self.algorithm_name} {self.task} in {idx+1} turn:\n Question:{question} \n Rag Output:{outputs} \n Answers: {eval_data[self.EvalData.InputStruction.answer]}')
                # calculate metric
                if self.task in ['ASQA','Factscore']:
                    # This two dataset need ALCE and Factscore to calculate the metrics
                    continue
                acc = self.EvalData.eval_acc(inference_results)
                EM = self.EvalData.eval_exact_match(inference_results)
                f1_score = self.EvalData.eval_f1_score(inference_results)
                self.print_fn(f'{self.algorithm_name} {self.task} in {idx+1} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
            # --> end of for loop
            self.EvalData.save_result(inference_results)
            if self.task in ['ASQA','Factscore']:
                return 'Inference completion'
            else:
                eval_result = {'Accuracy':acc, 'Exact match': EM, 'F1 score':f1_score}
                self.EvalData.save_evaluation_results(eval_result)
                return eval_result
        else:
            raise ModeNotFoundError("Mode must be interact or evaluation. Please provide a valid mode.")

    def infer(self, query: str)->tuple[str,dict[str,Any]]:
        '''
        infer function of naive rag
        '''
        generation_track = {}
        if self.realtime_retrieval:
            passages = self.retrieval.search(query) #self.retrieval.search(query) -> dict[int,dict]
            passages = self._truncate_passages(passages)
            collated_passages = self.collate_passages(passages)
            target_instruction = self.find_algorithm_instruction('Naive_rag', self.task)
            input = target_instruction.format_map({'passages': collated_passages, 'query': query})
            generation_track['cited passages'] = passages
        else:
            target_instruction = self.find_algorithm_instruction('Naive_rag-without_retrieval', self.task)
            input = target_instruction.format_map({'query': query})
        outputs_list = self.llm.generate(input) # llm.generate() -> list[BaseOutputs] so you have to get the text from BaseOutputs.text
        Outputs = outputs_list[0]
        outputs_text = Outputs.text
        generation_track['final answer'] = outputs_text
        return outputs_text, generation_track

    def steup_llm(self, args):
        if self.llm_mode == 'HF_Model':
            if self.use_vllm:
                llm = HF_VLLM(args)
                llm.load_model() # load_model() will load local model and tokenizer  
            else:
                llm = HF_Model(args)
                llm.load_model() # load_model() will load local model and tokenizer
        elif self.llm_mode == "Lora_Model":
            llm = Lora_Model(args)
            llm.load_model() #  load_model() will load base model and lora adapter then merged by peft to get complete model
        elif self.llm_mode == 'Openai_api':
            llm = OpenaiModel(args)
            llm.load_model() # load_model() will load api configs and tiktoken
        else:
            raise LanguageModelError("Language model must be huggingface or openai api.")
        return llm
    
    def setup_retrieval(self, args):
        if 'colbert' == self.retrieval_name:
            retrieval_model = ColbertRetrieve(args)
            retrieval_model.setup_retrieve()
        elif 'contriever' == self.retrieval_name:
            retrieval_model = ContrieverRrtieve(args)
            retrieval_model.setup_retrieve()
        elif  'colbert_api' == self.retrieval_name:
            retrieval_model = ColbertApi(args)
            retrieval_model.setup_retrieve()
        elif 'pregiven_passages' == self.retrieval_name:
            retrieval_model = None # no need setup retrieval model when pre-given passages prepared
        else:
            raise RetrievalModelError("invalid retrieval model")
        return retrieval_model 
    
    def collate_passages(self, passages:dict[int, Optional[dict]])-> str:
        collate = ''
        for rank_id, doc in passages.items(): 
            if doc is None:
                continue
            if 'title' in doc:
                collate += f'#Passages{rank_id}: ' '##Title: '+ doc['title'] + ' ##Content: ' + doc['text'] +'\n' 
            else:
                collate += f'#Passages{rank_id}: ' + doc['text'] +'\n'
        return collate

    def _truncate_passages(self, passages: dict[int, dict]) -> dict[int, dict]:
        '''
        The passages provided by wiki2023 contain an average of 700 words
        If n_doc set to 10, then 7000 words passages clearly exceed the LLM's window length
        As a result, adding _truncate_passages limits the length of each passage
        '''
        truncated_passages = {}
        for rank, passage_dict in passages.items():
            truncated_passages[rank] = {
                'id': passage_dict['id'],
                'title': passage_dict['title'],
                'text': self.truncate_text(passage_dict['text'], self.passages_max_length),
                'score': passage_dict['score']
            }
        return truncated_passages

    def truncate_text(self, passages:str, max_words:int)->str:
        words = passages.split() # "Split the passages into a list of words by space"
        if max_words == -1:
            truncated_words = words
        else:
            truncated_words = words[:max_words] # "Take the first max_words words"
        truncated_passages = ' '.join(truncated_words) # "Join the list of words back into a string"
        return truncated_passages

    def setup_logger(self, args):
        # set logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG) # Set the log level to DEBUG
        # build file_name based on args
        self.time = datetime.now().strftime('%m%d_%H%M_%S')
        if args.llm_mode == 'HF_Model' or args.llm_mode == 'Lora_Model':
            model_name = os.path.basename(self.llm_path.rstrip('/'))
            dir_name = args.algorithm_name + '-' + args.task + '-' + model_name + '-' + args.retrieval_name + '-' + f'{self.time}'
        else:
            dir_name = args.algorithm_name + '-' + args.task + '-' + args.llm_name + '-' + args.retrieval_name + '-' + f'{self.time}'
        self.output_dir = os.path.join(args.output_dir, args.task, dir_name)
        while os.path.exists(self.output_dir):
            # When multiple parallel processes attempt to create folders simultaneously, 
            # it triggers a random waiting mechanism for renaming to prevent file conflicts.
            print(f'{RED}file confliction, Re-generate the file{END}')
            random_wait = random.uniform(1,2)
            time.sleep(random_wait)
            self.time = datetime.now().strftime('%m%d_%H%M_%S')
            if args.llm_mode == 'HF_Model' or args.llm_mode == 'Lora_Model':
                model_name = os.path.basename(self.llm_path.rstrip('/'))
                dir_name = args.algorithm_name + '-' + args.task + '-' + model_name + '-' + args.retrieval_name + '-' + f'{self.time}'
            else:
                dir_name = args.algorithm_name + '-' + args.task + '-' + args.llm_name + '-' + args.retrieval_name + '-' + f'{self.time}'
            self.output_dir = os.path.join(args.output_dir, args.task, dir_name)
        # -> end of dir exists
        os.makedirs(self.output_dir)
        print(f'{GREEN}create log file success:{self.output_dir}{END}')
        if args.llm_mode == 'HF_Model' or args.llm_mode == 'Lora_Model':
            model_name = os.path.basename(self.llm_path.rstrip('/'))
            self.file_name = args.algorithm_name + '|' + args.task + '|' + model_name + '|' + args.retrieval_name + '|'
        else:
            self.file_name = args.algorithm_name + '|' + args.task + '|' + args.llm_name + '|' + args.retrieval_name + '|'
        # Create log file handler
        log_file = 'rag_output-' + self.file_name + f'time={self.time}.log'
        log_file = os.path.join(self.output_dir, log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Set log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger.info
    # TODO change name -> algorithm instruction 
    def find_algorithm_instruction(self, algorithm_name:str, dataset_name:str) -> str:
        target_instruction = ''
        for instruction in ALGORITHM_INSTRUCTIONS:
            if instruction['algorithm_name'] == algorithm_name and instruction['dataset_name'] == dataset_name:
                target_instruction = instruction['instruction']
                break
        if target_instruction == '':
            raise InstructionNotFoundError('Instruction name not recognized. Please provide a valid instruction key.')
        return target_instruction

# custom Exceptions
class ModeNotFoundError(Exception):
    pass

class InstructionNotFoundError(Exception):
    pass

class LanguageModelError(Exception):
    pass

class RetrievalModelError(Exception):
    pass

class GPUSETTINGERROR(Exception):
    pass