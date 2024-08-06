export CUDA_VISIBLE_DEVICES=7
python -i ./raglab/language_model/vllm_server.py\
    --config ./config/llm_server/llm_server.yaml