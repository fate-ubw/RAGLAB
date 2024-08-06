export CUDA_VISIBLE_DEVICES=1
python ./raglab/retrieval/colbert_api/colbert_server.py \
    --config ./config/colbert_server/colbert_server.yaml
