import os
import requests
import sys
import time
import logging
from tqdm import tqdm
from typing import Union
import numpy as np

class UnifiedApiRequest:
    class Outputs:
        def __init__(self):
            self.text = ""
            self.tokens_ids = None
            self.tokens_num = None
            self.tokens = None
            self.tokens_logprob = None
            self.tokens_prob = None
            self.cumulative_logprob = None
            self.logprobs = None
            self.text_logprobs = None

    def __init__(self, args):
        self.generation_stop = args.generation_stop if args.generation_stop != '' else None
        self.llm_name = args.llm_name
        self.api_key_path = args.api_key_path
        self.api_base = args.api_base
        self.local_proxy_url = args.local_proxy_url
        self.generate_maxlength = args.generate_maxlength
        self.temperature = args.temperature
        self.top_p = args.top_p

    def load_model(self):
        key_path = self.api_key_path
        assert os.path.exists(key_path), f"Please place your API Key in {key_path}."
        with open(key_path, 'r') as f:
            self.api_key = f.readline().strip()
        print(f'API base: {self.api_base}')
        print(f"API key: {self.api_key}")
        print(f"Local proxy URL: {self.local_proxy_url}")

    def generate(self, inputs: Union[str, list[str]]) -> list[Outputs]:
        if isinstance(inputs, str):
            inputs = [inputs]
        apioutputs_list = []
        for input_text in tqdm(inputs, desc="Generating outputs"):
            message = [{"role": "user", "content": input_text}]
            response = self.call_API(message)
            
            apioutput = self.Outputs()
            apioutput.text = response.get('data', '')
            apioutputs_list.append(apioutput)
        
        return apioutputs_list

    def call_API(self, message, max_retries=10):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.api_key
        }
        api_data = {
            "name": "test_model1",
            "inputs": {
                "stream": False,
                "msg": message[0]['content']
            },
            "temperature": self.temperature,
            "model": self.llm_name,
            "max_tokens": self.generate_maxlength
        }

        proxy_data = {
            "target_url": f'{self.api_base}',
            "headers": headers,
            "payload": api_data
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.local_proxy_url, json=proxy_data)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logging.error(f"API error: {e}. Attempt {attempt + 1} of {max_retries}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logging.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logging.critical("Max retries reached. Unable to get response from API.")
                    raise

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.generation_stop = ''
            self.llm_name = "gpt-4-1106-preview"
            self.api_key_path = "/home/wyd/zxw-cache/api_key.txt"
            self.api_base = "https://llmaiadmin-test.classba.cn/api/chat/call"
            self.local_proxy_url = "http://localhost:4998/api/proxy"
            self.generate_maxlength = 2048
            self.temperature = 0.9
            self.top_p = 1.0

    args = Args()
    unified_api = UnifiedApiRequest(args)
    unified_api.load_model()
    
    inputs = ["Please calculate 狄利克雷积分"]
    outputs = unified_api.generate(inputs)
    for output in outputs:
        print(output.text)