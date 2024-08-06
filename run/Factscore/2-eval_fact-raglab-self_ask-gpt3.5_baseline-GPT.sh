# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/self_ask-Factscore-gpt-3.5-turbo-colbert_api-0528_0254_28/rag_output-self_ask|Factscore|gpt-3.5-turbo|colbert_api|time=0528_0254_28.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose