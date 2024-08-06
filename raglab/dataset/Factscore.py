import os
import jsonlines
from raglab.dataset.PopQA import  PopQA
from datetime import datetime
from dataclasses import dataclass

class Factscore(PopQA):
    def __init__(self, args):
        super().__init__(args)

    @dataclass
    class InputStruction:
        question:str = 'input'
        answer:str = 'output'
        topic:str = 'topic'
        category = 'cat'

    @dataclass
    class OutputStruction:
        question:str = 'input'
        answer:str = 'answers'
        generation:str = 'output'
        topic:str = 'topic'
        category = 'cat'
        intermediate = 'intermediate'

    def record_result(self, eval_data, final_prediction_with_citation, inference_results, catation_docs=None, response_id=None, generation_track=None):
        if catation_docs is None and response_id is None and generation_track is None:
            inference_results.append(
                {
                    self.OutputStruction.question: eval_data[self.InputStruction.question], 
                    self.OutputStruction.answer: None ,
                    self.OutputStruction.generation: final_prediction_with_citation, 
                    self.OutputStruction.topic: eval_data[self.InputStruction.topic],
                    self.OutputStruction.category: eval_data[self.InputStruction.category], 
                }) 
        else:
            postprocessed_result = final_prediction_with_citation[response_id]
            inference_results.append(
                {
                    self.OutputStruction.question: eval_data[self.InputStruction.question], 
                    self.OutputStruction.answer: None ,
                    self.OutputStruction.generation: postprocessed_result, 
                    self.OutputStruction.topic: eval_data[self.InputStruction.topic],
                    self.OutputStruction.category: eval_data[self.InputStruction.category], 
                    self.OutputStruction.intermediate: generation_track["original_splitted_sentences"][response_id]
                }) 
        return inference_results
