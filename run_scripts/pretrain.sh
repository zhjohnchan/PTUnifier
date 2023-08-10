export TOKENIZERS_PARALLELISM=true

num_gpus=4
per_gpu_batchsize=32

python main.py \
  with seed=0 data_root=data/pretrain_arrows/ \
  num_gpus=${num_gpus} num_nodes=1 \
  task_pretrain_ptunifier \
  per_gpu_batchsize=${per_gpu_batchsize} \
  clip16 text_roberta \
  image_size=288 max_text_len=48 \
  tokenizer=downloaded/biomed_roberta_base
