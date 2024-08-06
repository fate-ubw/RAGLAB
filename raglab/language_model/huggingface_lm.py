from typing import  Union
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from raglab.language_model.base_lm import BaseLM
import pdb

class HF_Model(BaseLM):
    def __init__(self,args):
        super().__init__(args)
        self.llm_path = args.llm_path
        self.dtype = args.dtype
        self.generation_stop = args.generation_stop
        self.use_chat_template = args.use_chat_template

    def load_model(self):
        if self.dtype == 'half' or self.dtype == 'float16':
            self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path, device_map="auto", torch_dtype=torch.float16)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, skip_special_tokens=False, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.llm.eval()

    def generate(self, inputs: Union[str,list[str]])->list[BaseLM.Outputs]:
        if isinstance(inputs,str):
            inputs = [inputs]
        outputs_list = []
        for prompt in tqdm(inputs, desc="Generating outputs"):
            if self.use_chat_template is True:
                messages = [{"role": "user", "content": prompt}]
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").cuda(self.llm.device)
            else:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda(self.llm.device)
            instruction_len = input_ids.shape[1]
            if self.temperature > 0 or self.top_p <1:
                # nuelcus
                hf_outputs = self.llm.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature = self.temperature,
                    top_p = self.top_p,
                    max_length=instruction_len + self.generate_maxlength,
                    eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id = self.tokenizer.eos_token_id
                )
            else:
                # greedy 
                hf_outputs = self.llm.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_length=instruction_len + self.generate_maxlength,
                    eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id = self.tokenizer.eos_token_id
                )
            Outputs = self.Outputs()
            Outputs.tokens_ids = hf_outputs.sequences[0][instruction_len:].tolist()
            Outputs.tokens_num = len(Outputs.tokens_ids)
            text = self.tokenizer.decode(Outputs.tokens_ids, skip_special_tokens = False)
            # replace special tokens
            if "<|eot_id|>" in text or "<|end_of_text|>":
                text =  text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            else:
                text =  text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").eplace("<|end_of_text|>", "").strip()
            if '</s>' in text:
                text =  text.replace("<s> ", "").replace("</s>", "").strip()
            else:
                text =  text.replace("<s> ", "").strip()
            Outputs.text = text
            # calculate the probs of each tokens
            tokens_prob = []
            tokens_logprob = []
            logprobs = []
            for idx, token_id in enumerate(Outputs.tokens_ids): # attention the type of token_id is torch.tensor()
                token_logprob = hf_outputs.scores[idx].log_softmax(-1)[0][token_id].item()
                token_prob = hf_outputs.scores[idx].log_softmax(-1).exp()[0][token_id].item() # `outputs.scores` only records the logits of the generated tokens, so its length is equal to `generation_maxlength`.
                logprob_dict = {int(i):float(logprob) for i, logprob in enumerate(hf_outputs.scores[idx].log_softmax(-1)[0].tolist())}
                tokens_prob.append(token_prob)
                tokens_logprob.append(token_logprob)
                logprobs.append(logprob_dict)
            Outputs.tokens_logprob = tokens_logprob
            Outputs.tokens_prob = tokens_prob
            Outputs.cumulative_logprob = float(np.prod(Outputs.tokens_prob) / max(len(Outputs.tokens_prob), 1))
            Outputs.logprobs = logprobs
            outputs_list.append(Outputs)
        # --> end of for loop
        return outputs_list