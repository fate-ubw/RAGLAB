# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/iter_retgen-Factscore-gpt-3.5-turbo-colbert_api-0528_0229_02/rag_outputter_retgen|Factscore|gpt-3.5-turbo|colbert_api|time=0528_0229_02.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose