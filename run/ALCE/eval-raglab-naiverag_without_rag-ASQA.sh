# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/naive_rag-ASQA-Llama3-8B-baseline-colbert-0528_0206_53/rag_output-naive_rag|ASQA|Llama3-8B-baseline|colbert|time=0528_0206_53.jsonl' \
    --mauve \
    --qa