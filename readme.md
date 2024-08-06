# RAGLAB: A Modular and Research-Oriented Unified Framework for Retrieval-Augmented Generation

- RAGLAB is a modular, research-oriented open-source framework for Retrieval-Augmented Generation (RAG) algorithms. It offers reproductions of 6 existing RAG algorithms and a comprehensive evaluation system with 10 benchmark datasets, enabling fair comparisons between RAG algorithms and easy expansion for efficient development of new algorithms, datasets, and evaluation metrics.




![figure-1](./figures/Raglab-figure-1_00.png)

# 🌟Features
- **Comprehensive RAG Ecosystem:** Supports the entire RAG pipeline from data collection and training to auto-evaluation.
- **Advanced Algorithm Implementations:** Reproduces 6 state-of-the-art RAG algorithms, with an easy-to-extend framework for developing new algorithms.
- **Fair Comparison Platform:** Provides benchmark results for 6 algorithms across 5 task types and 10 datasets.
- **Efficient Retriever Client:** Offers local API for parallel access and caching, with average latency under 1 second.
- **Versatile Generator Support:** Compatible with 70B+ models, VLLM, and quantization techniques.
- **Flexible Instruction Lab:** Customizable instruction templates for various RAG scenarios.



# 🔨Install environment
- dev environment：pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
- [install miniconda](https://docs.anaconda.com/free/miniconda/index.html)

- git clone raglab
  ~~~bash
  git clone https://github.com/fate-ubw/raglab-exp.git
  ~~~
- create environment from yml file 
  ~~~bash
  cd raglab-exp
  conda env create -f environment.yml
  ~~~
- install flash-attn, en_core_web_sm, punkt manually
  ~~~bash
  pip install flash-attn==2.2
  python -m spacy download en_core_web_sm
  python -m nltk.downloader punkt
  ~~~

# 🤗 Model
- raglab need llama2-7b, llama3-8b, colbertv2.0, selfrag_llama2_7b
  ~~~bash
  cd raglab-exp
  mkdir model
  cd model
  mkdir output_models
  mkdir Llama-2-7b-hf
  huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b-hf/
  mkdir Meta-Llama-3-8B
  huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir Meta-Llama-3-8B/
  mkdir Meta-Llama-3-70B
  huggingface-cli download meta-llama/Meta-Llama-3-70B --local-dir Meta-Llama-3-70B/
  mkdir selfrag_llama2_7b
  huggingface-cli download selfrag/selfrag_llama2_7b --local-dir selfrag_llama2_7b/
  mkdir colbertv2.0
  huggingface-cli download colbert-ir/colbertv2.0 --local-dir colbertv2.0/
  ~~~

# 💽 process wiki2023 as vector database

## 10-samples test
- 10-samples test is aimed at validating the environment
- run colbert embedding process enwiki-20230401-10samples.tsv
  1. Change root path for variables: `checkpoint`, `index_dbPath`, `collection` in
[wiki2023-10samples_tsv-2-colbert_embedding.py](https://github.com/fate-ubw/raglab-exp/blob/main/preprocess/colbert-wiki2023-preprocess/wiki2023-db_into_tsv-10samples.py). In file paths, colbert encounters many issues when using relative paths to generate embeddings. Therefore, the current version of raglab uses absolute paths. 
  ~~~bash
    # change root path
  checkpoint = '/your_root_path/raglab-exp/model/colbertv2.0'
  index_dbPath = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023-10samples'
  collection = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv'
  ~~~
  2. run
  ~~~bash
  cd raglab-exp
  sh run/wiki2023_preprocess/2-wiki2023-10samples_tsv-2-colbert_embedding.sh
  ~~~
- Embedding precess will take around 15mins in first time.
- The first time colbert processes embeddings, it takes a relatively long time because it needs to recompile the `torch_extensions`. However, calling the processed embeddings does not require a long time. If there are no errors and the retrieved text can be printed, it indicates that the environment is correct.
## Run Raglab with 10-samples embedding
- test selfrag  base on 10-samples embedding
- After processing with colbert embeddings, you can start running the algorithms in raglab. All algorithms integrated in raglab include two modes: `interact` and `evaluation`. The test stage demonstrates in `interact` mode, just for fun 🤗.
- Modify the `index_dbPath` and `text_dbPath` in config file:[selfrag_reproduction-interact-short_form-adaptive_retrieval.yaml](https://github.com/fate-ubw/raglab-exp/blob/main/config/selfrag_reproduction/selfrag_reproduction-interact-short_form-adaptive_retrieval.yaml)
  ~~~bash
  index_dbPath: /your_root_path/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023-10samples
  text_dbPath: /your_root_path/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv
  ~~~
- run [selfrag](https://arxiv.org/abs/2310.11511) (short form & adaptive retrieval) interact mode test 10-samples embedding
  ~~~bash
  cd raglab-exp
  sh run/rag_inference/3-selfrag_reproduction-interact-short_form-adaptive_retrieval.sh
  ~~~
- Congratulations！！！Now you have already know how to run raglab 🌈
- In raglab, each algorithm has 10 queries built-in in interact mode which are sampled from benchmark


## embedding whole wiki2023
- you can download the [colbert embdding wiki2023]() as raglab database(40Gb)
~~~bash
cd /raglab-exp/data/retrieval/colbertv2.0_embedding
gdown --id xxxxxx
# unzip commend for 
~~~
- modify the path in meta.json file
- embedding whole wiki2023 to vector need 22 hours, so we recommend download prepared embedding

### download wiki2023 raw data
- current version of raglab use wiki2023 as database
- we get source wiki2023 get from [factscore](https://github.com/shmsw25/FActScore)
  - method1: url for download wiki2023:[google_drive](https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view) 
  - method2: install throuth gdown 
  ~~~bash
  cd raglab-exp/data/retrieval/colbertv2.0_passages
  mkdir wiki2023
  pip install gdown
  gdown --id 1mekls6OGOKLmt7gYtHs0WGf5oTamTNat
  ~~~

### preprocess wiki2023
- If the 10-samples test is passed successfully, you can proceed with processing wiki2023.
1. preprocess `.db -> .tsv` (Colbert can only read files in .tsv format.)
    ~~~bash
    cd raglab-exp
    sh run/wiki2023_preprocess/3-wiki2023_db-2-tsv.sh
    ~~~
2. `.tsv -> embedding`
  - remember to change the root  path of `checkpoint`, `index_dbPath` and `collection`
    ~~~bash
      # change root path
        checkpoint = '/your_root_path/raglab-exp/model/colbertv2.0'
        index_dbPath = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023-10samples'
        collection = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv'
    ~~~
  - run bash script
    ~~~bash
    cd raglab-exp
    sh run/wiki2023_preprocess/4-wiki2023_tsv-2-colbert_embedding.sh
    ~~~



# 💽 Process wiki2018 as vector database
- This section is a tutorial on using wiki2018

## Download text files
  - Directly download wiki2018 raw database using wget
~~~bash
cd raglab-exp/data/retrieval/colbertv2.0_passages/wiki2018
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
~~~

## Process raw wiki2018 into colbert format

~~~bash
cd raglab-exp
sh run/wiki2018_preprocess/1-wiki2018_tsv_2_tsv.sh
~~~

## Modify wiki2018 embedding config file
1. Change the path
~~~
cd /raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2018/indexes/wiki2018
vim metadata.json 
~~~
- You only need to modify two paths in the metadata.json file. Here, simply delete the original paths and copy the following paths. Other parameters do not need to be modified.
~~~sh
"collection": "/home/ec2-user/SageMaker/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2018/wiki2018.tsv",
"experiment": "/home/ec2-user/SageMaker/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2018",
~~~
- After modification, you can directly start the colbert server. For experimental startup method, refer to the last section of the readme: Inference experiments.



# Inference experiments
## Retrieval server & api
- The inference experiments require running hundreds of scripts in parallel. If each script loads the wiki2023 database separately, not only does it require a large amount of RAM, but loading the wiki2023 database each time also takes a considerable amount of time, which is a significant waste of computing resources. Therefore, RagLab has designed [colbert server & colbert api](https://github.com/fate-ubw/raglab-exp/tree/main/raglab/retrieval/colbert_api) to address the problem of multi-task parallel retrieval. By runnging local colbert server, tasks can call the colbert api to obtain retrieval results, greatly reducing the inference time for multiple tasks.
- Attention: colbert_server need atleast 60GB ram 
  ~~~bash
  cd raglab-exp
  sh run/colbert_server/colbert_server.sh
  ~~~
- open another terminal test your ColBERT server
~~~bash
cd raglab-exp
sh run/colbert_server/ask_api.sh
~~~
- ColBERT server started successfully!!! 🌈
## Automatic GPU Scheduler
- inference experiments require running hundreds of scripts in parallel, the [automatic gpu scheduler](https://github.com/ExpectationMax/simple_gpu_scheduler) needs to be used to automatically allocate GPUs for different bash scripts in Parallel.
- install `simple_gpu_scheduler`
  ~~~bash
  pip install simple_gpu_scheduler
  ~~~
- run hundreds of experiments in one line 😎
  ~~~bash
  cd raglab-exp
  simple_gpu_scheduler --gpus 0,1,2,3,4,5,6,7 < auto_gpu_scheduling_scripts/auto_run_scripts-jeff.py
  ~~~
- how to write your_script.txt?
  - here is an example
  ~~~bash
  # auto_inference_selfreg-7b.txt
  sh run/rag_inference/selfrag_reproduction/selfrag_reproduction-evaluation-short_form-PubHealth-adaptive_retrieval-pregiven_passages.sh
  sh run/rag_inference/selfrag_reproduction/selfrag_reproduction-evaluation-short_form-PubHealth-always_retrieval-pregiven_passages.sh
  ~~~



## Fine tune llama3 & self rag 
- The base models for raglab baseline and selfrag use llama3-instruction-8b. Since selfrag was further fine-tuned on additional data during the fine-tuning stage, in order to make a fair comparison, the baseline model also needs to be fine-tuned.
### download self rag train data
- we get the train data from [selfrag](https://github.com/AkariAsai/self-rag/tree/main)
- google drive [url](https://drive.google.com/file/d/10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk/view)
- download through gdown
  ~~~bash
  cd raglab-exp/data/train_data/
  gdown --id 10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk
  ~~~
### 10-samples test for fintune
- The 10-samples train dataset has been processed, please directly start the bash script to begin testing.
- Note: The test script only uses one GPU
  - full weight requires 80GB VRam GPU
  ~~~bash
  cd raglab-exp
  sh run/rag_train/script_finetune-llama3-baseline-full_weight-10samples.sh
  ~~~
  - LoRA (Low-Rank Adaptation) requires at least 26GB of VRAM
  ~~~bash
  cd raglab-exp
  sh run/rag_train/script_finetune-llama3-baseline-Lora-10samples.sh
  ~~~
- Congratulations！！！You can now start fine-tuning the baseline and selfrag-8b🤖
## finetune self rag 8b
- full weight finetune
  ~~~bash
  cd raglab-exp
  sh run/rag_train/script_finetune-selfrag_8b-full_weight.sh
  ~~~
- lora finetune 
  ~~~bash
  cd raglab-exp
  sh run/rag_train/script_finetune-selfrag_8b-Lora.sh
  ~~~
## finetune llama3-8b as baseline
- preprocess train data. Train data for baseline model need remove special tokens.
  ~~~bash
  cd raglab-exp
  sh run/traindataset_preprocess/selfrag_traindata-remove_special_tokens.sh
  ~~~
- then you will get baseline train_data without special token and passages (Q: what is specal token? Anawer: special tokens is a concept proposed by SelfRAG)
- full weight finetune llama3-8b-baseline ues processed data
  ~~~bash
  sh run/rag_train/script_finetune-llama3-baseline-full_weight.sh
  ~~~
- lora finetune llama3-8b-baseline
  ~~~bash
  cd raglab-exp
  sh run/rag_train/script_finetune-llama3-baseline-Lora.sh
  ~~~
## Lora finetune llama3-70b as baseline
- preprocess train data. Train data for baseline model need remove special tokens.
  ~~~bash
  cd raglab-exp
  sh run/traindataset_preprocess/selfrag_traindata-remove_special_tokens.sh
  ~~~
- lora finetune llama3-70b-baseline ues processed data
  ~~~bash
  sh run/rag_train/script_finetune-llama3-70B-baseline-Lora.sh
  ~~~

## QLora finetune llama3-70B as baseline
- preprocess train data. Train data for baseline model need remove special tokens.
  ~~~bash
  cd raglab-exp
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


## :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).