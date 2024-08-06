from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
class BaseLM(ABC):
    def __init__(self,args) -> None:
        self.temperature = args.temperature
        self.generate_maxlength = args.generate_maxlength
        self.top_p = args.top_p

    @dataclass
    class Outputs:
        '''
        Outputs struction unify all kinds of output of openai api
        '''
        text: Optional[str] = None      # generation text
        tokens_ids: Optional[list[int]] = None # generation token id 
        tokens_num: Optional[int] = None  # len of generation tokens
        tokens: Optional[str] = None 
        tokens_prob: Optional[list[float]]= None # generation token probs 
        tokens_logprob: Optional[list[float]]= None # generation token probs
        cumulative_logprob: Optional[int] = None  
        logprobs: Optional[list[dict[int, float]]] = None # vocabulary table of generation token
        text_logprobs: Optional[list[dict[str,float]]] = None # only openai api get this args
        def __repr__(self):
            return (
                "Outputs(\n"
                f"    text={self.text},\n"
                f"    tokens_ids={self.tokens_ids},\n"
                f"    tokens_num={self.tokens_num},\n"
                f"    tokens={self.tokens},\n"
                f"    tokens_prob={self.tokens_prob},\n"
                f"    tokens_logprob={self.tokens_logprob},\n"
                f"    cumulative_logprob={self.cumulative_logprob}\n"
                ")"
            )

    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate(self): 
        pass