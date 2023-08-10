export TOKENIZERS_PARALLELISM=true

load_path=downloaded/ptunifier.ckpt
pseudo_vision_token_pool_size=2048
pseudo_langauge_token_pool_size=2048
seed=0

# == 1. Uni-modal Tasks ==
# Vision-Only
for data_frac in 0.01 0.1 1.; do
  # 1.1 Multi-label Classification on CheXpert
  num_gpus=4
  per_gpu_batchsize=4
  python main.py with seed=${seed} data_root=data/finetune_vision_arrows/ \
    num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
    task_finetune_mlc_chexpert clip16 text_roberta \
    image_size=384 clip_resizedcrop tokenizer=downloaded/biomed_roberta_base \
    pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
    pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
    load_path=${load_path} \
    data_frac=${data_frac}

  # 1.2 Classification on RNAS Pneumonia
  num_gpus=4
  per_gpu_batchsize=4
  python main.py with seed=${seed} data_root=data/finetune_vision_arrows/ \
    num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
    task_finetune_mlc_pnsa_pneumonia clip16 text_roberta \
    image_size=384 clip_resizedcrop tokenizer=downloaded/biomed_roberta_base \
    pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
    pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
    load_path=${load_path} \
    data_frac=${data_frac}
done

# Language Only
# 1.3 Classification on RadNLI
num_gpus=4
per_gpu_batchsize=4
python main.py with seed=${seed} data_root=data/finetune_language_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_nli_radnli_plus clip16 text_roberta \
  image_size=384 clip_resizedcrop tokenizer=downloaded/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path}

# 1.4 Radiology Report Summarization on MIMIC-CXR
num_gpus=4
per_gpu_batchsize=16
python main.py with seed=${seed} data_root=data/finetune_vision_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_clm_mimic_cxr_language_only clip16 text_roberta \
  image_size=384 clip_resizedcrop tokenizer=downloaded/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path}

# == 2. Cross-modal Tasks ==
# Image-to-Text Retrieval
# 2.1 Cross-modal Retrieval on ROCO (Zero-shot)
num_gpus=4
per_gpu_batchsize=4
python main.py with seed=${seed} data_root=data/finetune_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_irtr_roco get_recall_metric=True \
  clip16 text_roberta \
  image_size=288 clip_resizedcrop tokenizer=downloaded/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path} \
  test_only=True

# 2.2 Cross-modal Retrieval on ROCO (Fine-tuned)
num_gpus=4
per_gpu_batchsize=4
python main.py with seed=${seed} data_root=data/finetune_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_irtr_roco get_recall_metric=False \
  clip16 text_roberta \
  image_size=384 clip_resizedcrop tokenizer=downloaded/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path}

# Image-to-Text Generation
# 2.3 Report Report Generation on MIMIC-CXR
num_gpus=4
per_gpu_batchsize=8
python main.py with seed=${seed} data_root=data/finetune_vision_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_clm_mimic_cxr_vision_only clip16 text_roberta \
  image_size=384 clip_resizedcrop \
  tokenizer=downloaded/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path}

# == 3. Multi-modal Tasks ==
# Visual Question Answering
# 3.1 Visual Question Answering on VQA-RAD
num_gpus=4
per_gpu_batchsize=16
python main.py with seed=${seed} data_root=data/finetune_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_vqa_vqa_rad clip16 text_roberta \
  image_size=384 tokenizer=downloaded/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path}

# 3.2 Visual Question Answering on SLACK
num_gpus=4
per_gpu_batchsize=8
python main.py with seed=${seed} data_root=data/finetune_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_vqa_slack clip16 text_roberta \
  image_size=384 clip_resizedcrop tokenizer=downloaded/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path}

# 3.3 Visual Question Answering on MedVQA-2019
num_gpus=4
per_gpu_batchsize=8
python main.py with seed=${seed} data_root=data/finetune_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_vqa_medvqa_2019 clip16 text_roberta \
  image_size=384 clip_resizedcrop tokenizer=downloaded/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path}

# Image-and-Text-to-Text Generation
# 3.4 Multi-modal Radiology Report Summarization on MIMIC-CXR
num_gpus=4
per_gpu_batchsize=8
python main.py with seed=${seed} data_root=data/finetune_vision_arrows/ \
  num_gpus=${num_gpus} per_gpu_batchsize=${per_gpu_batchsize} num_nodes=1 \
  task_finetune_clm_mimic_cxr_vision_language clip16 text_roberta \
  image_size=384 clip_resizedcrop \
  tokenizer=downloaded/biomed_roberta_base \
  pseudo_vision_token_pool_size=${pseudo_vision_token_pool_size} \
  pseudo_langauge_token_pool_size=${pseudo_langauge_token_pool_size} \
  load_path=${load_path}
