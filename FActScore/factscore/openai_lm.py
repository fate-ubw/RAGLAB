from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging
import pdb
class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        openai.api_key = api_key.strip()
        openai.api_base = "https://api2.aigcbest.top/v1" #  api_base
        print(f'current api base: {openai.api_base}')
        print(f"Current API: {api_key.strip()}")
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=4096, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(message, temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            output = response["choices"][0]["message"]["content"]
            return output, response
        elif self.model_name == "InstructGPT":
            # Call API
            response = call_GPT3(prompt, temp=self.temp)
            # Get the output from the response
            output = response["choices"][0]["text"]
            return output, response
        else:
            raise NotImplementedError()

def call_ChatGPT(message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False):
    # call GPT-3 API until result is provided and then return it
    print("Calling GPT-3.5")
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        print(message)
        try:
            response = openai.ChatCompletion.create(model=model_name,
                                                    messages=message,
                                                    max_tokens=max_len,
                                                    temperature=temp)
            received = True
        except:

            num_rate_errors += 1
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                assert False
            logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(prompt, model_name="text-davinci-003", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
    # call GPT-3 API until result is provided and then return it
    print("text-davinci-003")
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.Completion.create(model=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo) 
            received = True
        except:
            print('error of openai request')
            error = sys.exc_info()[0]
            print(f'error type:{error}')
            num_rate_errors += 1
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API error: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response

if __name__ == "__main__":
    openai.api_key = "sk-AsnWDR9ki4r7SIH5B2E40dF4586248649376635a3aEf8d2b"
    openai.api_base = "https://api.aigcbest.top/v1"
    pdb.set_trace()
    print(call_GPT3([{"role": "user", "content": "Give me an English name."}]))