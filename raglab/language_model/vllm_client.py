import requests
from typing import Union, List, Optional, Dict
from vllm import SamplingParams
from transformers import AutoTokenizer
import pdb
import time
# from raglab.language_model.base_lm import BaseLM

class VLLM_Client():

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
    
    def __init__(self, args=None) -> None:
        self.url = "http://localhost:5001"
        self.critic_path = args.critic_path 
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.critic_path, skip_special_tokens=False, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, input: Union[str, List[str]], sampling_params: SamplingParams=None):
        endpoint = f"{self.url}/api/generate"
        # Extract all parameters from SamplingParams
        if sampling_params is not None:
            params = {
                "n": sampling_params.n,
                "best_of": sampling_params.best_of,
                "presence_penalty": sampling_params.presence_penalty,
                "frequency_penalty": sampling_params.frequency_penalty,
                "repetition_penalty": sampling_params.repetition_penalty,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "min_p": sampling_params.min_p,
                "use_beam_search": sampling_params.use_beam_search,
                "length_penalty": sampling_params.length_penalty,
                "early_stopping": sampling_params.early_stopping,
                "stop": sampling_params.stop,
                "stop_token_ids": sampling_params.stop_token_ids,
                "include_stop_str_in_output": sampling_params.include_stop_str_in_output,
                "ignore_eos": sampling_params.ignore_eos,
                "max_tokens": sampling_params.max_tokens,
                "logprobs": sampling_params.logprobs,
                "prompt_logprobs": sampling_params.prompt_logprobs,
                "skip_special_tokens": sampling_params.skip_special_tokens,
                "spaces_between_special_tokens": sampling_params.spaces_between_special_tokens
            }
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            payload = {
                "input": input,
                "params": params  # Convert SamplingParams to a dictionary
            }
        else:
            payload = {
                "input": input
            }
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            # Convert the JSON response back to a list of Output objects
            return self.dicts_to_outputs(response.json())
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def output_to_dict(output: 'VLLM_Client.Outputs') -> Dict:
        return {
            "text": output.text,
            "tokens_ids": output.tokens_ids,
            "tokens_num": output.tokens_num,
            "tokens": output.tokens,
            "tokens_prob": output.tokens_prob,
            "tokens_logprob": output.tokens_logprob,
            "cumulative_logprob": output.cumulative_logprob,
        }

    @staticmethod
    def dict_to_output(data: Dict) -> 'VLLM_Client.Outputs':
        output = VLLM_Client.Outputs()
        for key, value in data.items():
            setattr(output, key, value)
        return output

    @classmethod
    def outputs_to_dicts(cls, outputs: List['VLLM_Client.Outputs']) -> List[Dict]:
        return [cls.output_to_dict(output) for output in outputs]

    @classmethod
    def dicts_to_outputs(cls, data: List[Dict]) -> List['VLLM_Client.Outputs']:
        return [cls.dict_to_output(item) for item in data]

if __name__ == "__main__":
    class Args():
        def __init__(self) -> None:
            self.url = "http://localhost:5001"
            self.critic_path = './model/output_models/unified-Critic-8B-baseline_2w/'
    args = Args()
    client = VLLM_Client(args)
    client.load_model()  # This doesn't do anything but is here for consistency
    # Example input
    start_time = time.time()
    input_text = "how do you do?"
    # Example SamplingParams with all possible parameters
    result = client.generate(input_text)
    end_time = time.time()
    print(f'use time:{end_time-start_time}')
    print(result[0])