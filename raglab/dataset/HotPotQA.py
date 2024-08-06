import pdb
from raglab.dataset.PopQA import PopQA
import numpy as np
from raglab.dataset.metrics import HotPotF1
from dataclasses import dataclass


class HotPotQA(PopQA):
    def __init__(self, args):
        super().__init__(args)

    @dataclass
    class InputStruction:
        question:str = 'question'
        answer:str = 'answer'

    @dataclass
    class OutputStruction:
        question:str = 'question'
        answer:str = 'answer'
        generation:str = 'generation'
    
    def eval_f1_score(self, infer_results: list[dict]) -> float:
        '''
        the HotpotQA need to preprocess specific cases for 'yes', 'no', and 'noanswer' predictions.
        '''
        print('Start calcualte F1 score!')
        eval_reaults = []
        for _, data in enumerate(infer_results):
            if type(data[self.OutputStruction.answer]) is str:
                answer = [data[self.OutputStruction.answer]]
            elif type(data[self.OutputStruction.answer]) is list:
                answer = data[self.OutputStruction.answer]
            else:
                raise InvalidAnswerType("The type of answer is invalid. Only str and list[str] is valid. Check the answer in your raw data.")
            
            metric_result = HotPotF1(data[self.OutputStruction.generation], answer)
            eval_reaults.append(metric_result)
        # TODO 这里应该把结果存储下来***.json.eval_result 
        return float(np.mean(eval_reaults))

class InvalidAnswerType(Exception):
    pass