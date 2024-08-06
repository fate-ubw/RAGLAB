from raglab.dataset.PopQA import  PopQA
from dataclasses import dataclass
import pdb

class ASQA(PopQA):
    def __init__(self, args):
        super().__init__(args)

    @dataclass
    class InputStruction:
        '''
        The goal of constructing InputStruction and OutputStruction is to achieve the separation of algorithm logic and data, 
        so that users only need to add new dataset structures according to the rules without modifying the algorithm logic.
        '''
        question:str = 'question'
        qa_pairs = 'qa_pairs'
        answer:str = 'answer'
        pregiven_passages:str = 'docs'

    @dataclass
    class OutputStruction:
        question:str = 'question'
        answer:str = 'answer'
        generation:str = 'output'
        cite_passages:str = 'docs'
        generation_track:str = 'intermediate'

    def save_result(self, inference_result: list[dict])-> None:
        '''
        the format of ASQA if based on ALCE 
        '''
        new_results = [{"data": inference_result, "args": [], "total_cost": 0.0, "azure_filter_fail": ""}]
        super().save_result(new_results)

    def record_result(self, eval_data, final_prediction_with_citation, inference_results, catation_docs = None, response_id = None, generation_track = None):
        '''
        record inference results
        '''
        eval_data.pop('wikipages', None)
        eval_data.pop('sample_id', None)
        eval_data.pop('docs', None)
        # eval_data
        if catation_docs is None and response_id is None and generation_track is None:
            # for other algorithm
            eval_data[self.OutputStruction.generation] = final_prediction_with_citation
            inference_results.append(eval_data)
        elif generation_track is not None and "original_splitted_sentences" in generation_track:
            # for self rag
            eval_data[self.OutputStruction.generation] = final_prediction_with_citation[response_id]
            eval_data[self.OutputStruction.cite_passages] = catation_docs[response_id]
            eval_data[self.OutputStruction.generation_track] = generation_track['original_splitted_sentences'][response_id]
            inference_results.append(eval_data)
        else:
            # for self rag
            eval_data[self.OutputStruction.generation] = final_prediction_with_citation[response_id]
            eval_data[self.OutputStruction.cite_passages] = catation_docs[response_id]
            inference_results.append(eval_data)
        return inference_results