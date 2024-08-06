# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/selfrag_reproduction-Factscore-selfrag_llama3_8b-epoch_0_1-colbert_api-0528_0241_31/rag_output-selfrag_reproduction|Factscore|selfrag_llama3_8b-epoch_0_1|colbert_api|time=0528_0241_31.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose