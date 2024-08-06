export CUDA_VISIBLE_DEVICES=0
python ./raglab/retrieval/colbert_api/colbert_server.py \
    --config ./config/colbert_server/colbert_server-10samples.yaml