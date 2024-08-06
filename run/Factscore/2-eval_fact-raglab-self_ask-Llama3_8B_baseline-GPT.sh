# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/self_ask-Factscore-Llama3-8B-baseline-colbert_api-0528_0336_33/rag_output-self_ask|Factscore|Llama3-8B-baseline|colbert_api|time=0528_0336_33.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose