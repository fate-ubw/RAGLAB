# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/query_rewrite_rag-ASQA-Llama3-70B-baseline-adapter-colbert_api-0610_2016_20/rag_output-query_rewrite_rag|ASQA|Llama3-70B-baseline-adapter|colbert_api|time=0610_2016_20.jsonl' \
    --mauve \
    --qa