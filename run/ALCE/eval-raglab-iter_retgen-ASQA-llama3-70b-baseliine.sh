# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/iter_retgen-ASQA-Llama3-70B-baseline-adapter-colbert_api-0611_0236_15/rag_output-iter_retgen|ASQA|Llama3-70B-baseline-adapter|colbert_api|time=0611_0236_15.jsonl' \
    --mauve \
    --qa