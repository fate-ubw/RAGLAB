
from typing import Union, List
import argparse
from ruamel.yaml import YAML
import numpy as np
from dataclasses import dataclass
from typing import Optional
from flask import Flask, request, jsonify
from functools import lru_cache
from vllm import LLM, SamplingParams

def over_write_args_from_file(args, yml):
    """
    overwrite arguments according to config file
    """
    if yml == '':
        return
    yaml = YAML(typ='rt') # rt is (round-trip) mode
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read())
        for k in dic:
            setattr(args, k, dic[k])


class VLLMServer():

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

    def __init__(self, args):
        self.temperature = args.temperature
        self.generate_maxlength = args.generate_maxlength
        self.top_p = args.top_p
        self.llm_path = args.llm_path
        self.dtype = args.dtype
        self.generation_stop = args.generation_stop
        self.include_stop_token = args.include_stop_token
        self.use_chat_template = args.use_chat_template
        self.counter = 0
        self.app = Flask(__name__)
        self.load_model()  # Call the parent's load_model method
        self.port = args.port
        self.app.add_url_rule("/api/generate", view_func=self.api_generate, methods=["POST"])

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

    def generate(self, inputs: Union[str,list[str]], sampling_params = None)->list[Outputs]:
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

    def run(self):
        self.app.run("0.0.0.0", self.port)

    def api_generate(self):
        if request.method == "POST":
            self.counter += 1
            print(f"The {self.counter}th API call succeeded.")
            data = request.json
            return self.process_generate_request(data.get("input"), data.get("params"))
        else:
            return ('', 405)
        
    def process_generate_request(self, input: Union[str, list[str]], params: dict = None):
        if params:
            sampling_params = SamplingParams(**params)
        else:
            sampling_params = None
        
        outputs = self.generate(input, sampling_params)
        
        results = []
        for output in outputs:
            result = {
                "text": output.text,
                "tokens_ids":output.tokens_ids,
                "tokens_num": output.tokens_num,
                "tokens": output.tokens,
                "tokens_logprob": output.tokens_logprob,
                "tokens_prob": output.tokens_prob,
                "cumulative_logprob": output.cumulative_logprob,
                "logprobs":output.logprobs,
                "text_logprobs": output.text_logprobs
            }
            results.append(result)

        return jsonify(results)

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_path", type=str, help="Path to the language model")
    parser.add_argument("--dtype", type=str, help="Data type for the model")
    parser.add_argument("--generation_stop", type=str, default="", help="Stop token for generation")
    parser.add_argument("--include_stop_token", type=bool, default=False, help="Include stop token in output")
    parser.add_argument("--use_chat_template", type=bool, default=False, help="Use chat template")
    parser.add_argument("--port", type=int, default=5001, help="Port for the server")
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    args = parser.parse_args()
    over_write_args_from_file(args, args.config)
    return args

if __name__ == "__main__":
    args = get_config()
    server = VLLMServer(args)
    server.run()