# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/active_rag-Factscore-Llama3-70B-baseline-adapter-colbert_api-0611_1558_40/rag_output-active_rag|Factscore|Llama3-70B-baseline-adapter|colbert_api|time=0611_1558_40.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose