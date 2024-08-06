from raglab.dataset.PopQA import  PopQA
from raglab.dataset.PubHealth import PubHealth
from dataclasses import dataclass

class StrategyQA(PubHealth):
    def __init__(self, args):
        super().__init__(args)

    @dataclass
    class InputStruction:
        question:str =  'question'
        answer:str = 'answer'

    @dataclass
    class OutputStruction:
        question:str = 'question'
        answer:str = 'answer' 
        generation:str = 'generation'