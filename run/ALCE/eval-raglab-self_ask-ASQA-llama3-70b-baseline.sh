# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/self_ask-ASQA-Llama3-70B-baseline-adapter-colbert_api-0611_0921_05/rag_output-self_ask|ASQA|Llama3-70B-baseline-adapter|colbert_api|time=0611_0921_05.jsonl' \
    --mauve \
    --qa