# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/query_rewrite_rag-Factscore-Llama3-70B-baseline-adapter-colbert_api-0610_2015_53/rag_output-query_rewrite_rag|Factscore|Llama3-70B-baseline-adapter|colbert_api|time=0610_2015_53.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose