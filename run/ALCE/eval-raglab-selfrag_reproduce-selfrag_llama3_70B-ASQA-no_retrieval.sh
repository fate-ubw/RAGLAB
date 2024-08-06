# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/selfrag_reproduction-ASQA-selfrag_llama3_70B-adapter-colbert_api-0612_1053_34/rag_output-selfrag_reproduction|ASQA|selfrag_llama3_70B-adapter|colbert_api|time=0612_1053_34.jsonl' \
    --mauve \
    --qa