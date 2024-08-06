# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/query_rewrite_rag-ASQA-gpt-3.5-turbo-colbert_api-0528_0213_15/rag_output-query_rewrite_rag|ASQA|gpt-3.5-turbo|colbert_api|time=0528_0213_15.jsonl' \
    --mauve \
    --qa
