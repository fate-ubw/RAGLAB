import os
import openai
import sys
import time
import logging
from tqdm import tqdm
from typing import Union, Tuple
import re
import numpy as np
from raglab.language_model.base_lm import BaseLM
import tiktoken
import pdb

sensitive_words = ["淫"]

class OpenaiModel(BaseLM):
    def __init__(self,args):
        super().__init__(args)
        self.generation_stop = args.generation_stop
        if self.generation_stop == '':
            self.generation_stop = None
        self.llm_name = args.llm_name
        self.api_key_path = args.api_key_path
        self.api_base = args.api_base
        self.api_logprobs = args.api_logprobs
        self.api_top_logprobs = args.api_top_logprobs

    def load_model(self):
        # load api key
        key_path = self.api_key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        openai.api_key = api_key.strip()
        openai.api_base = self.api_base
        print(f'current api base: {openai.api_base}')
        print(f"Current API: {api_key.strip()}")
        self.tokenizer = tiktoken.encoding_for_model(self.llm_name)

    def generate(self, inputs: Union[str,list[str]])-> list[BaseLM.Outputs]:
        '''
        Current version of OpenaiModel batch inference was not implemented
        '''
        if isinstance(inputs,str):
            inputs = [inputs]
        apioutputs_list = []
        for input_text in tqdm(inputs, desc="Generating outputs"):
            # original_text = input_text
            # input_text, is_modified = self.remove_sensitive_words(input_text)
            
            # if is_modified:
            #     logging.info(f"Remove sensitive text! origanl text: {original_text[:50]}... revised text: {input_text[:50]}...")
            
            message = [{"role": "user", "content": input_text}]
            if self.api_logprobs is False:
                response = self.call_ChatGPT(message, model_name=self.llm_name, max_len=self.generate_maxlength, temp=self.temperature, top_p=self.top_p, stop = self.generation_stop)
                # collate Apioutputs
                if response is None:
                    continue
                apioutput = self.Outputs()
                apioutput.text = response["choices"][0]["message"]["content"]
                apioutput.tokens_ids = self.tokenizer.encode(apioutput.text)
                apioutput.prompt_tokens_num = response["usage"]["prompt_tokens"]
                apioutput.tokens_num = len(apioutput.tokens_ids)
                apioutputs_list.append(apioutput)
            else:
                for i in range(1,50): # max time of recall is 10 times
                    print(f'The {i}-th API call')
                    response = self.call_ChatGPT(message, model_name=self.llm_name, max_len=self.generate_maxlength, temp=self.temperature, top_p=self.top_p, stop = self.generation_stop, logprobs=self.api_logprobs, top_logprobs=self.api_top_logprobs)
                    # collate Apioutputs
                    if response is None:
                        continue
                    if 'logprobs' in response["choices"][0]:
                        if response["choices"][0]['logprobs'] is not None:
                            apioutput = self.Outputs()
                            apioutput.text = response["choices"][0]["message"]["content"]
                            apioutput.tokens_ids = self.tokenizer.encode(apioutput.text)
                            apioutput.tokens_num = len(apioutput.tokens_ids)
                            apioutput.prompt_tokens_num = response["usage"]["prompt_tokens"]
                            apioutput.tokens = [content['token'] for content in response["choices"][0]['logprobs']['content']]
                            apioutput.tokens_logprob = [content['logprob'] for content in response["choices"][0]['logprobs']['content']]
                            apioutput.tokens_prob = np.exp(apioutput.tokens_logprob).tolist()
                            apioutput.cumulative_logprob = float(np.prod(apioutput.tokens_prob) / max(len(apioutput.tokens_prob), 1))
                            apioutput.logprobs = []
                            apioutput.text_logprobs = []
                            for content in response["choices"][0]['logprobs']['content']: # content:dict[token/logprobs/top_logprobs] 每个content都包含一个 token 的信息
                                top_logprobs = content['top_logprobs']
                                one_token_vocab = {}
                                text_token_vocab = {}
                                for log_prob in top_logprobs: # top_logprobs:list[dict[token/logprobs/bytes]]
                                    token_str = log_prob['token']
                                    try:
                                        token_id = self.tokenizer.encode_single_token(token_str)
                                    except KeyError:
                                        print(f"Token '{token_str}' not found in vocabulary")
                                        continue
                                    token_logprob = log_prob['logprob']
                                    one_token_vocab[token_id] = float(token_logprob)
                                    text_token_vocab[token_str] = float(token_logprob)
                                apioutput.logprobs.append(one_token_vocab)
                                apioutput.text_logprobs.append(text_token_vocab)
                            # end of for loop
                            apioutputs_list.append(apioutput)
                            print(f'API call success')
                            break
                        else:
                            pass # logprob is None so recall chatgpt in next turn
                    else:
                        pass
                # --> end of recall loop
            # --> end of else
        # --> end of main loop
        return apioutputs_list

    def call_ChatGPT(self,message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, top_p = 1.0, stop = None, logprobs = False, top_logprobs = None, verbose=False):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        num_rate_errors = 0
        while not received:
            try:
                response = openai.ChatCompletion.create(model=model_name,
                                                        messages=message,
                                                        max_tokens=max_len,
                                                        temperature=temp,
                                                        top_p = top_p,
                                                        stop = stop,
                                                        logprobs = logprobs,
                                                        top_logprobs = top_logprobs,
                                                        seed = 2024)
                received = True
            except:
                num_rate_errors += 1
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                    return None
                logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
                time.sleep(np.power(2, num_rate_errors))
        return response


    def remove_sensitive_words(self, text: str) -> Tuple[str, bool]:
        original_text = text
        for word in sensitive_words:
            text = re.sub(word, '', text)
        return text, text != original_text