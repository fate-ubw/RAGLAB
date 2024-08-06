# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/naive_rag-ASQA-Llama3-70B-baseline-adapter-colbert-0610_1611_23/rag_output-naive_rag|ASQA|Llama3-70B-baseline-adapter|colbert|time=0610_1611_23.jsonl' \
    --mauve \
    --qa