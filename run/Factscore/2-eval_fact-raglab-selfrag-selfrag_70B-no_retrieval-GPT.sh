# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/selfrag_reproduction-Factscore-selfrag_llama3_70B-adapter-colbert_api-0614_1307_49/rag_output-selfrag_reproduction|Factscore|selfrag_llama3_70B-adapter|colbert_api|time=0614_1307_49.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose