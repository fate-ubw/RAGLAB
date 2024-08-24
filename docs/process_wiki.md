# process knowledge database into vector database
## 💽 process wiki2023 as vector database

### 10-samples test
- 10-samples test is aimed at validating the environment
- run colbert embedding process enwiki-20230401-10samples.tsv
  1. Change root path for variables: `checkpoint`, `index_dbPath`, `collection` in
[wiki2023-10samples_tsv-2-colbert_embedding.py](./preprocess/colbert-wiki2023-preprocess/wiki2023-10samples_tsv-2-colbert_embedding.py). Colbert enforces the use of absolute paths, so you need to modify the paths for the following three variables
  ~~~bash
    # change root path
  checkpoint = '/your_root_path/RAGLAB/model/colbertv2.0'
  index_dbPath = '/your_root_path/RAGLAB/data/retrieval/colbertv2.0_embedding/wiki2023-10samples'
  collection = '/your_root_path/RAGLAB/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv'
  ~~~
  2. run
  ~~~bash
  cd RAGLAB
  sh run/wiki2023_preprocess/2-wiki2023-10samples_tsv-2-colbert_embedding.sh
  ~~~
- Embedding precess will take around 15mins in first time.
- The first time colbert processes embeddings, it takes a relatively long time because it needs to recompile the `torch_extensions`. However, calling the processed embeddings does not require a long time. If there are no errors and the retrieved text can be printed, it indicates that the environment is correct.

###  Wiki2023 raw data source
- we get source wiki2023 get from [factscore](https://github.com/shmsw25/FActScore)
- **Note**: RAGLAB already provides enwiki2023 source data on HuggingFace, so there's no need to download it again. This information is just to provide the source of the data.
  - download method: install throuth gdown 
  ~~~bash
  cd RAGLAB/data/retrieval/colbertv2.0_passages
  mkdir wiki2023
  pip install gdown
  gdown --id 1mekls6OGOKLmt7gYtHs0WGf5oTamTNat
  ~~~

### preprocess wiki2023
- If the 10-samples test is passed successfully, you can proceed with processing wiki2023.
1. preprocess `.db -> .tsv` (Colbert can only read files in .tsv format.)
    ~~~bash
    cd RAGLAB
    sh run/wiki2023_preprocess/3-wiki2023_db-2-tsv.sh
    ~~~
2. `.tsv -> embedding`
  - remember to change the root  path of `checkpoint`, `index_dbPath` and `collection`
    ~~~bash
      vim preprocess/colbert-wiki2023-preprocess/wiki2023_tsv-2-colbert_embedding.py
      # change root path
        checkpoint = '/your_root_path/RAGLAB/model/colbertv2.0'
        index_dbPath = '/your_root_path/RAGLAB/data/retrieval/colbertv2.0_embedding/wiki2023-10samples'
        collection = '/your_root_path/RAGLAB/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv'
    ~~~
  - run bash script
    ~~~bash
    cd RAGLAB
    sh run/wiki2023_preprocess/4-wiki2023_tsv-2-colbert_embedding.sh
    ~~~
  - This usually takes about 20 hours, depending on your computer's performance

## 💽 Process wiki2018 as vector database
- This section is a tutorial on using wiki2018

### wiki2018 raw data source
  - we get source wiki2018 get from [DPR](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz)
  - Directly download wiki2018 raw database using wget
~~~bash
cd RAGLAB/data/retrieval/colbertv2.0_passages/wiki2018
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
~~~

### Porcess wiki2018
1. tsv -> tsv
  ~~~bash
  cd RAGLAB
  sh run/wiki2018_preprocess/1-wiki2018_tsv_2_tsv.sh
  ~~~
2. tsv -> embedding
  ~~~bash
  cd RAGLAB
  sh run/wiki2018_preprocess/2-wiki2018_tsv-2-colbert_embedding.sh
  ~~~
