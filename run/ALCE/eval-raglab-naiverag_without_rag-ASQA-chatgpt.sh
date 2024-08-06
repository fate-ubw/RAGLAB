# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/naive_rag-ASQA-gpt-3.5-turbo-colbert-0528_0155_36/rag_output-naive_rag|ASQA|gpt-3.5-turbo|colbert|time=0528_0155_36.jsonl' \
    --mauve \
    --qa