from typing import Optional, Any
from tqdm import tqdm
import pdb
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag

class QueryRewrite_rag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)

    def infer(self, query:str)->tuple[str, dict[str,Any]]:
        '''
        infer function of rrr
        paper:[https://arxiv.org/abs/2305.14283]
        source code: [https://github.com/xbmxb/RAG-query-rewriting/tree/main]
        '''
        # rewrite the query
        generation_track = {}
        target_instruction = self.find_algorithm_instruction('query_rewrite_rag-rewrite', self.task)
        query_with_instruction = target_instruction.format_map({'query':query})
        rewrite_query = self._rewrite(query_with_instruction)
        generation_track['rewrite query'] = rewrite_query
        # retrieval
        passages = self.retrieval.search(rewrite_query)
        passages = self._truncate_passages(passages)
        generation_track['cited passages'] = passages
        collated_passages = self.collate_passages(passages)
        target_instruction = self.find_algorithm_instruction('query_rewrite_rag-read', self.task)
        query_with_instruction = target_instruction.format_map({'query':query, 'passages':collated_passages})
        # read
        output_list = self.llm.generate(query_with_instruction)
        Output = output_list[0]
        output_text = Output.text
        generation_track['final answer'] = output_text
        return output_text, generation_track

    def _rewrite(self, query):
        output_list = self.llm.generate(query)
        Output = output_list[0]
        rewrite_query = Output.text
        return rewrite_query

