from typing import Optional
import pdb
import spacy
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
from raglab.language_model import BaseLM
class ActiveRag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        self.init(args)

    def init(self, args):
        self.filter_prob = args.filter_prob
        self.masked_prob = args.masked_prob
        self.max_fianl_answer_length = args.max_fianl_answer_length
        self.nlp = spacy.load("en_core_web_sm")

    def infer(self, query:str)->tuple[str,dict]:
        next_iter_flag = True
        answer_len = 0
        final_generation = ''
        iter_step = 1
        generation_track = {} # TODO: active rag generation track
        generation_track[iter_step] = {'instruction': None, 'retrieval_input': query, 'passages':None, 'generation':None}
        print(f'source question -> {query}')
        while next_iter_flag and answer_len < self.max_fianl_answer_length:
            if iter_step == 1:
                retrieval_input = query
                passages = self.retrieval.search(retrieval_input)
                passages = self._truncate_passages(passages)
            collated_passages = self.collate_passages(passages) 
            target_instruction = self.find_algorithm_instruction('active_rag-read', self.task)
            inputs = target_instruction.format_map({'passages': collated_passages, 'query': query})
            inputs = inputs + final_generation 
            # get look_ahead
            output_list = self.llm.generate(inputs)
            Outputs = output_list[0]
            print(f'whole look ahead -> {Outputs.text}')
            if len(Outputs.text)==0:
                break
            # get first sentence from look_ahead
            look_ahead = self._truncate_text(Outputs)
            print(f'look ahead -> {look_ahead.text}')
            # mask low prob tokens in look_ahead
            masked_look_ahead = self._mask_lowprob_tokens(look_ahead)
            if len(masked_look_ahead.tokens_ids) > 0:
                # re-retrieve passages based on look_ahead
                print(f'retrieval input/masked look ahead -> { masked_look_ahead.text}')
                # retrieval
                passages = self.retrieval.search(masked_look_ahead.text)
                passages = self._truncate_passages(passages)
                collated_passages = self.collate_passages(passages)
                target_instruction = self.find_algorithm_instruction('active_rag-read', self.task)
                inputs = target_instruction.format_map({'passages': collated_passages, 'query': query})
                # concatenate instruction + question + least answer
                inputs = inputs + final_generation 

                outputs_list = self.llm.generate(inputs)
                Outputs = outputs_list[0]

                print(f'final outpus -> {Outputs.text}')
            else:
                # If no low prob tokens in look_ahead, look_ahead is the current turn outputs
                Outputs = look_ahead
            if len(Outputs.text) == 0:
                break
            # get the first sentence from outputs 
            Doc = self.nlp(Outputs.text)
            first_sentence = list(Doc.sents)[0].text
            final_generation += ' ' + first_sentence
            print(f'final generation -> {final_generation}')
            # clculate the len of current generation length
            truncated_outputs_id = self.llm.tokenizer.encode(first_sentence)
            answer_len +=  len(truncated_outputs_id)
            number_of_sents = len(list(Doc.sents))
            # Decide continue or not.If the final_outputs contains more than one sentence, the next_iter_flag will set True
            if number_of_sents > 1:
                next_iter_flag = True
            else:
                next_iter_flag = False
            iter_step += 1
        # end of while
        return final_generation, generation_track

    def _truncate_text(self, llm_outputs:BaseLM.Outputs)->BaseLM.Outputs: 
        '''
        '''
        Doc = self.nlp(llm_outputs.text)
        first_sent = list(Doc.sents)[0].text
        first_sent_tokenid = self.llm.tokenizer.encode(first_sent)
        first_sent_len = len(first_sent_tokenid)
        first_sent_prob = llm_outputs.tokens_prob[0:first_sent_len]
        return BaseLM.Outputs(text=first_sent, tokens_ids=first_sent_tokenid, tokens_prob=first_sent_prob)
        

    def _mask_lowprob_tokens(self, llm_outputs:BaseLM.Outputs)->BaseLM.Outputs:
        '''
        raglab rerpoduce the Masked sentences as implicit queries in active rag algorithm(https://arxiv.org/abs/2305.06983)
        '''
        masked_text = ''
        masked_tokens_ids = []
        masked_tokens_prob = []
        filered_prob = [prob for prob in llm_outputs.tokens_prob if prob < self.filter_prob]
        if len(filered_prob)>0:
            for token_id, token_prob in zip(llm_outputs.tokens_ids, llm_outputs.tokens_prob):
                if token_prob > self.masked_prob:
                    masked_tokens_ids.append(token_id)
                    masked_tokens_prob.append(token_prob)
            masked_text = self.llm.tokenizer.decode(masked_tokens_ids)
        # end of if
        if '</s>' in masked_text:
            masked_text =  masked_text.replace("<s> ", "").replace("</s>", "").strip()
        else:
            masked_text =  masked_text.replace("<s> ", "").strip()
        return BaseLM.Outputs(text=masked_text, tokens_ids=masked_tokens_ids, tokens_prob=masked_tokens_prob)
