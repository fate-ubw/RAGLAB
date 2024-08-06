from raglab.dataset.PopQA import  PopQA
from dataclasses import dataclass

class WikiMultiHopQA(PopQA):
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
