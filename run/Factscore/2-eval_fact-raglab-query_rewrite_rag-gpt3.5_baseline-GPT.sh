# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/query_rewrite_rag-Factscore-gpt-3.5-turbo-colbert_api-0528_0212_20/rag_output-query_rewrite_rag|Factscore|gpt-3.5-turbo|colbert_api|time=0528_0212_20.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose