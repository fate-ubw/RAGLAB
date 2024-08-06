export CUDA_VISIBLE_DEVICES=0
export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python  ./preprocess/colbert-wiki2023-preprocess/wiki2023_tsv-2-colbert_embedding.py
