from typing import Union
import numpy as np
from vllm import LLM, SamplingParams
from raglab.language_model.base_lm import BaseLM
import pdb

class HF_VLLM(BaseLM):
    def __init__(self, args):
        super().__init__(args)
        self.llm_path = args.llm_path
        self.dtype = args.dtype
        self.generation_stop = args.generation_stop
        self.include_stop_token = args.include_stop_token
        self.use_chat_template = args.use_chat_template

    def load_model(self):
        self.llm = LLM(model=self.llm_path, tokenizer=self.llm_path, dtype=self.dtype)
        self.tokenizer = self.llm.get_tokenizer()
        vocab_size = self.tokenizer.vocab_size
        special_token_size = len(self.tokenizer.added_tokens_decoder)
        # remove redundancy special tokens
        real_special_tokens = [{idx:token} for idx,token in self.tokenizer.added_tokens_decoder.items() if idx >= vocab_size]
        special_token_size = len(real_special_tokens)
        # In current version of vllm llama3 need add eos_token_id & "<|eot_id|>" for stop generation

        if self.generation_stop != '':
            self.sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, 
                                                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")], stop = [self.generation_stop], include_stop_str_in_output = self.include_stop_token,
                                                repetition_penalty= 1, max_tokens = self.generate_maxlength, logprobs=vocab_size + special_token_size, skip_special_tokens = False)
        else:
            self.sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, 
                                                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")] , include_stop_str_in_output = self.include_stop_token,
                                                repetition_penalty= 1, max_tokens = self.generate_maxlength, logprobs=vocab_size + special_token_size, skip_special_tokens = False)

    def generate(self, inputs: Union[str,list[str]], sampling_params = None)->list[BaseLM.Outputs]:
        if isinstance(inputs,str):
            inputs = [inputs]
        outputs_list = []
        # add chat template
        inputs_with_chat_template = []
        if self.use_chat_template is True:
            # llama2 & llama3-instruction need add chat template to get the best performance
            for input in inputs:
                inputs_with_chat_template.append(self.tokenizer.apply_chat_template([{'role': 'user', 'content': input}], add_generation_prompt=True, tokenize=False))
            inputs = inputs_with_chat_template            
        if sampling_params is None:
            outputs = self.llm.generate(inputs, self.sampling_params)
        else:
            outputs = self.llm.generate(inputs, sampling_params)
        for RequestOutput in outputs:
            Outputs = self.Outputs()
            text = RequestOutput.outputs[0].text
            # replace eos bos
            if "<|eot_id|>" in text or "<|end_of_text|>":
                text =  text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            else:
                text =  text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").eplace("<|end_of_text|>", "").strip()
            if '</s>' in text:
                text =  text.replace("<s> ", "").replace("</s>", "").strip()
            else:
                text =  text.replace("<s> ", "").strip()
            Outputs.text = text
            Outputs.tokens_ids = RequestOutput.outputs[0].token_ids
            Outputs.cumulative_logprob = RequestOutput.outputs[0].cumulative_logprob
            Outputs.tokens_num = len(Outputs.tokens_ids)
            # tokens_prob & tokens_logprob
            Outputs.tokens_logprob = [logprob[token_id] for token_id, logprob in zip(Outputs.tokens_ids, RequestOutput.outputs[0].logprobs)]
            Outputs.tokens_prob = np.exp(Outputs.tokens_logprob).tolist()   
            Outputs.logprobs = RequestOutput.outputs[0].logprobs
            outputs_list.append(Outputs)
        # --> end of for loop
        return outputs_list