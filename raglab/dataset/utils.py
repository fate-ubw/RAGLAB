import os
from datetime import datetime
import jsonlines
import json
from ruamel.yaml import YAML
import pdb
# TASK_LIST = ['PopQA','PubHealth','ArcChallenge', 'TriviaQA', 'ASQA', 'Factscore', 
#              'HotPotQA', 'StrategyQA', '2WikiMultiHopQA','Feverous', 'MMLU']

TASK_LIST = ['PopQA','PubHealth','Arc', 'TriviaQA', 'ASQA', 'Factscore', 
             'HotPotQA', 'StrategyQA', '2WikiMultiHopQA','Feverous', 'MMLU'] 
def load_jsonlines(file:str)-> list[dict]:
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst 

def get_dataset(args) -> object:
    # base class
    from raglab.dataset.base_dataset.MultiChoiceQA import MultiChoiceQA
    from raglab.dataset.base_dataset.QA import QA
    # advanced dataset class
    from raglab.dataset.PopQA import PopQA
    from raglab.dataset.PubHealth import PubHealth
    from raglab.dataset.ArcChallenge import ArcChallenge
    from raglab.dataset.TriviaQA import TriviaQA
    from raglab.dataset.HotPotQA import HotPotQA
    from raglab.dataset.ASQA import ASQA
    from raglab.dataset.Factscore import Factscore
    from raglab.dataset.StrategyQA import StrategyQA
    from raglab.dataset.WikiMultiHopQA import WikiMultiHopQA
    from raglab.dataset.Feverous import Feverous
    from raglab.dataset.MMLU import MMLU

    if 'PopQA' == args.task:
        EvalData = PopQA(args)
    elif 'PubHealth' == args.task:
        EvalData = PubHealth(args)
    elif 'Arc' == args.task:
        EvalData = ArcChallenge(args)
    elif 'TriviaQA' == args.task:
        EvalData = TriviaQA(args)
    elif 'ASQA' == args.task:
        EvalData = ASQA(args)
    elif 'Factscore' == args.task:
        EvalData = Factscore(args)
    elif 'HotPotQA' == args.task:
        EvalData = HotPotQA(args)
    elif 'StrategyQA' == args.task:
        EvalData = StrategyQA(args)
    elif '2WikiMultiHopQA' == args.task:
        EvalData = WikiMultiHopQA(args)
    elif 'Feverous' == args.task:
        EvalData = Feverous(args)
    elif 'MMLU' == args.task:
        EvalData = MMLU(args)
    else:
        raise TaskNotFoundError("Task not recognized. Please provide a valid args.task.")
    return EvalData

def get_args_form_config(yml)-> dict:
    """
    """
    # blacklist = ['num_gpu','eval_datapath', 'output_dir', 
    #              'index_dbPath', 'text_dbPath', 'retriever_modelPath', 
    #              'nbits', 'llm_mode', 'api_key_path', 'api_base','use_vllm','realtime_retrieval',
    #              'use_seed', 'seed']
    if yml == '':
        return
    yaml = YAML(typ='rt') # rt is (round-trip) mode
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read())
        # for k in dic:
        #     if k not in blacklist:
        #         if k == 'llm_path':
        #             dic[k] = os.path.basename(dic[k].rstrip('/'))
        #         file_name += f'{k}={str(dic[k])}|'
    return dic

class TaskNotFoundError(Exception):
    pass

if __name__ == "__main__":
    '''
    test program
    '''
    def find_yaml_files(folder_path: str) -> list:
        """
        Find all YAML files in the specified folder path.
        """
        yaml_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.yaml') or file.endswith('.yml'):
                    yaml_files.append(os.path.join(root, file))
        return yaml_files
 
    def main():
        # Test get_name_form_config function
        folder_path = '/home/wyd/raglab-exp/config/iterative_rag'  # Provide the path to your YAML file here
        yaml_files = find_yaml_files(folder_path)
        for yml_file in yaml_files:
            file_name = get_name_form_config(yml_file)
            print(f"Generated filename: {file_name}")
    # --> end of def
    main()