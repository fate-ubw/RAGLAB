# export CUDA_VISIBLE_DEVICES=1
python  ./FActScore/factscore/factscorer.py  \
    --input_path './data/eval_results/Factscore/naive_rag-Factscore-Llama3-70B-baseline-adapter-colbert_api-0610_1838_01/rag_output-naive_rag|Factscore|Llama3-70B-baseline-adapter|colbert_api|time=0610_1838_01.jsonl' \
    --model_name "retrieval+ChatGPT"\
    --openai_key ./api_keys.txt \
    --data_dir ./data/factscore \
    --verbose