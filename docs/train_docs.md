# 🤖 Train models

## 10-samples test for fintune
- The 10-samples train dataset has been processed, please directly start the bash script to begin testing.
- Note: The test script only uses one GPU
  - full weight requires 80GB VRam GPU
  ~~~bash
  cd RAGLAB
  sh run/rag_train/script_finetune-llama3-baseline-full_weight-10samples.sh
  ~~~
  - LoRA (Low-Rank Adaptation) requires at least 26GB of VRAM
  ~~~bash
  cd RAGLAB
  sh run/rag_train/script_finetune-llama3-baseline-Lora-10samples.sh
  ~~~
- Congratulations！！！You can now start fine-tuning the baseline model and selfrag-8B
## finetune self rag 8b
- full weight finetune
  ~~~bash
  cd RAGLAB
  sh run/rag_train/script_finetune-selfrag_8b-full_weight.sh
  ~~~
- lora finetune 
  ~~~bash
  cd RAGLAB
  sh run/rag_train/script_finetune-selfrag_8b-Lora.sh
  ~~~
## finetune llama3-8b as baseline
- preprocess train data. Train data for baseline model need remove special tokens.
  ~~~bash
  cd RAGLAB
  sh run/traindataset_preprocess/selfrag_traindata-remove_special_tokens.sh
  ~~~
- then you will get baseline train_data without special token and passages (Q: what is specal token? Anawer: special tokens is a concept proposed by SelfRAG)
- full weight finetune llama3-8b-baseline ues processed data
  ~~~bash
  sh run/rag_train/script_finetune-llama3-baseline-full_weight.sh
  ~~~
- lora finetune llama3-8b-baseline
  ~~~bash
  cd RAGLAB
  sh run/rag_train/script_finetune-llama3-baseline-Lora.sh
  ~~~
## Lora finetune llama3-70b as baseline
- preprocess train data. Train data for baseline model need remove special tokens.
  ~~~bash
  cd RAGLAB
  sh run/traindataset_preprocess/selfrag_traindata-remove_special_tokens.sh
  ~~~
- lora finetune llama3-70b-baseline ues processed data
  ~~~bash
  sh run/rag_train/script_finetune-llama3-70B-baseline-Lora.sh
  ~~~

## QLora finetune llama3-70B as baseline
- preprocess train data. Train data for baseline model need remove special tokens.
  ~~~bash
  cd RAGLAB
  sh run/traindataset_preprocess/selfrag_traindata-remove_special_tokens.sh
  ~~~
- 8bit QLora finetune llama3-70B 
  ~~~bash
  sh run/rag_train/script_finetune-llama3-70B-baseline-QLora-8bit.sh
  ~~~
- 4bit QLora fintune llama3-70B
  ~~~bash
  sh run/rag_train/script_finetune-llama3-70B-baseline-QLora-4bit.sh
  ~~~

## 8bit QLora finetune selfrag-70B as baseline
- 8bit Qlora finetune slefrag 70B
  ~~~bash
    sh run/rag_train/script_finetune-selfrag_llama3-70b-QLora-8bit.sh
  ~~~
- 4bit Qlora finetune slefrag 70B
  ~~~bash
    sh run/rag_train/script_finetune-selfrag_llama3-70b-QLora-4bit.sh
  ~~~