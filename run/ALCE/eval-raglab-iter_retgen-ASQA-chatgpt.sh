# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/iter_retgen-ASQA-gpt-3.5-turbo-colbert_api-0528_0235_14/rag_output-iter_retgen|ASQA|gpt-3.5-turbo|colbert_api|time=0528_0235_14.jsonl' \
    --mauve \
    --qa