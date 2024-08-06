# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/active_rag-Factscore-gpt-3.5-turbo-colbert_api-0528_0332_14/rag_output-active_rag|Factscore|gpt-3.5-turbo|colbert_api|time=0528_0332_14.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose