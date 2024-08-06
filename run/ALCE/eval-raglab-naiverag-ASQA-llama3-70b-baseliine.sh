# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/naive_rag-ASQA-Llama3-70B-baseline-adapter-colbert_api-0610_1910_46/rag_output-naive_rag|ASQA|Llama3-70B-baseline-adapter|colbert_api|time=0610_1910_46.jsonl' \
    --mauve \
    --qa