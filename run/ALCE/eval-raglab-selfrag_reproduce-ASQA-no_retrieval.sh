# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/selfrag_reproduction-ASQA-selfrag_llama3_8b-epoch_0_1-colbert_api-0528_0717_36/rag_output-selfrag_reproduction|ASQA|selfrag_llama3_8b-epoch_0_1|colbert_api|time=0528_0717_36.jsonl' \
    --mauve \
    --qa