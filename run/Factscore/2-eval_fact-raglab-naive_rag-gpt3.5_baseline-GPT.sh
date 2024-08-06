# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/naive_rag-Factscore-gpt-3.5-turbo-colbert_api-0528_0154_06/rag_output-naive_rag|Factscore|gpt-3.5-turbo|colbert_api|time=0528_0154_06.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose