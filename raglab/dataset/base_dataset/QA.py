from abc import ABC, abstractmethod

class QA(ABC):
    def __init__(self, args):
        self.output_dir =args.output_dir
        self.llm_path = args.llm_path
        self.eval_datapath = args.eval_datapath

    @abstractmethod
    def load_dataset(self): # The class that inherits the class must override load_dataset() methods
        pass
    
    @abstractmethod
    def save_result(self):# The class that inherits the class must override save_inference_result() methods
        pass

    def preprocess(self,input):
        return input