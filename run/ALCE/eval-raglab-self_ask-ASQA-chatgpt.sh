# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/self_ask-ASQA-gpt-3.5-turbo-colbert_api-0528_0258_57/rag_output-self_ask|ASQA|gpt-3.5-turbo|colbert_api|time=0528_0258_57.jsonl' \
    --mauve \
    --qa
