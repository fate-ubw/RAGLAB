# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/iter_retgen-Factscore-Llama3-8B-baseline-colbert_api-0528_0250_28/rag_outputter_retgen|Factscore|Llama3-8B-baseline|colbert_api|time=0528_0250_28.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose