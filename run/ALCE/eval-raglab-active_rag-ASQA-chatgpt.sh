# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/active_rag-ASQA-gpt-3.5-turbo-colbert_api-0528_0334_19/rag_output-active_rag|ASQA|gpt-3.5-turbo|colbert_api|time=0528_0334_19.jsonl' \
    --mauve \
    --qa