# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/self_ask-ASQA-Llama3-8B-baseline-colbert_api-0528_0337_00/rag_output-self_ask|ASQA|Llama3-8B-baseline|colbert_api|time=0528_0337_00.jsonl' \
    --mauve \
    --qa
