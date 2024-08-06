import pdb
from dataclasses import dataclass
from raglab.dataset.PubHealth import PubHealth

class MMLU(PubHealth):
    def __init__(self, args):
        super().__init__(args)

    @dataclass
    class InputStruction:
        question:str = 'question'
        answer:str = 'answerKey'
        choices:str = 'choices'

    @dataclass
    class OutputStruction:
        question:str = 'question'
        answer:str = 'answerKey'
        generation:str = 'generation'
    
    def preprocess(self, eval_data):
        choices = eval_data["choices"]
        postprocess_text = ''
        for answer_text, label in zip(choices['text'], choices['label']):
            postprocess_text += '\n'+ label + ': ' + answer_text
        eval_data[self.InputStruction.question] += postprocess_text
        return eval_data