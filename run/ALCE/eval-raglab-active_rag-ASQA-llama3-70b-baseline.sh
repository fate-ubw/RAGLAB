# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/active_rag-ASQA-Llama3-70B-baseline-adapter-colbert_api-0611_1318_13/rag_output-active_rag|ASQA|Llama3-70B-baseline-adapter|colbert_api|time=0611_1318_13.jsonl' \
    --mauve \
    --qa