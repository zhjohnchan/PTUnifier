from sacred import Experiment

ex = Experiment("METER", save_git_info=False)


def _loss_names(d):
    ret = {
        "mlm": 0,  # Pre-training: Masked Language Modeling
        "mim": 0,  # Pre-training: Masked Image Modeling
        "umlm": 0,  # Pre-training: Uni-modal Masked Language Modeling
        "umim": 0,  # Pre-training: Uni-modal Masked Image Modeling
        "itm": 0,  # Pre-training: Image-Text Matching
        "itc": 0,  # Pre-training: Contrastive Learning (Image-Text Contrastive)
        "vqa": 0,  # Fine-tuning: Visual Question Answering
        "cls": 0,  # Fine-tuning: Classification
        "mlc": 0,  # Fine-tuning: Multi-label Classification
        "irtr": 0,  # Fine-tuning: cross-modal retrieval
        "clm": 0,  # Fine-tuning: Causal Language Modeling
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "ptunifier"
    seed = 0
    datasets = ["medicat", "roco"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqa_label_size = 3129
    mlc_label_size = 14
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    draw_false_text = 0
    sent_level = False

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # MIM decoder Setting
    mim_prob = 0.75
    mim_decoder_hidden_size = 384
    mim_decoder_num_layers = 4
    mim_decoder_num_heads = 6
    norm_pix_loss = True
    mim_layer = -1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = None  # 100
    max_steps = None  # 100000
    warmup_steps = 10000
    end_lr = 0
    lr_multiplier_head = 5  # multiply lr for prediction heads
    lr_multiplier_multi_modal = 5  # multiply lr for the multi-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    default_root_dir = "checkpoints"

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0
    num_gpus = 8
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    # CLASSIFICATION SETTING
    label_column_name = ""
    label_size = {"i_meth": 85, "p_meth": 45, "i_meth_label": 15, "p_meth_label": 7,
                  "chexpert": 5, "pnsa_pneumonia": 1,
                  "chemprot": 6, "radnli_plus": 3}

    # Pseudo Token Pool
    pseudo_vision_token_pool_size = 2048
    pseudo_langauge_token_pool_size = 2048
    num_pseudo_vision_tokens = 49
    num_pseudo_langauge_tokens = 32

    # Contrastive Learning
    cl_temp = 0.1

    # For vision data
    data_frac = 1.
    vision_only = False

    # For language data
    language_only = False

    # For Causal Language Modeling
    clm_max_text_len = 256
    clm_do_sample = True
    clm_num_beams = 5


@ex.named_config
def task_pretrain_ptunifier():
    exp_name = "task_pretrain_ptunifier"
    datasets = ["medicat", "roco", "mimic_cxr"]
    loss_names = _loss_names({"itm": 0.5, "mlm": 1, "itc": 0.1})
    batch_size = 256
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 48
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["clip_resizedcrop"]
    val_transform_keys = ["clip"]
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_multiplier_head = 5
    lr_multiplier_multi_modal = 5
    num_top_layer = 6
    hidden_size = 768
    num_heads = 12

    precision = 16
    mim_layer = 3
    sent_level = True

    # Pseudo Token Pool
    pseudo_vision_token_pool_size = 2048
    pseudo_langauge_token_pool_size = 2048
    num_pseudo_vision_tokens = 49
    num_pseudo_langauge_tokens = 32

    # Contrastive Learning
    cl_temp = 0.1


@ex.named_config
def task_finetune_vqa_vqa_rad():
    exp_name = "task_finetune_vqa_vqa_rad"
    datasets = ["vqa_vqa_rad"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 64
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 32
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

    vqa_label_size = 498


@ex.named_config
def task_finetune_vqa_slack():
    exp_name = "task_finetune_vqa_slack"
    datasets = ["vqa_slack"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 32
    max_epoch = 15
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 32
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

    vqa_label_size = 222


@ex.named_config
def task_finetune_vqa_medvqa_2019():
    exp_name = "task_finetune_vqa_medvqa_2019"
    datasets = ["vqa_medvqa_2019"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 32
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 32
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

    vqa_label_size = 79


@ex.named_config
def task_finetune_irtr_roco():
    exp_name = "task_finetune_irtr_roco"
    datasets = ["irtr_roco"]
    loss_names = _loss_names({"irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_multiplier_head = 5
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


@ex.named_config
def task_finetune_mlc_chexpert():
    exp_name = "task_finetune_mlc_chexpert"
    datasets = ["mlc_chexpert"]
    loss_names = _loss_names({"mlc": 1})
    batch_size = 16
    max_epoch = 1
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 10
    tokenizer = "bert-base-uncased"
    max_text_len = 32
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

    label_column_name = "chexpert"
    vision_only = True


@ex.named_config
def task_finetune_mlc_pnsa_pneumonia():
    exp_name = "task_finetune_mlc_pnsa_pneumonia"
    datasets = ["mlc_psna_pneumonia"]
    loss_names = _loss_names({"mlc": 1})
    batch_size = 16
    max_epoch = 3
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 10
    tokenizer = "bert-base-uncased"
    max_text_len = 32
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

    label_column_name = "pnsa_pneumonia"
    vision_only = True


@ex.named_config
def task_finetune_clm_mimic_cxr_vision_only():
    exp_name = "task_finetune_clm_mimic_cxr_vision_only"
    datasets = ["clm_mimic_cxr"]
    loss_names = _loss_names({"clm": 1})
    batch_size = 128
    max_epoch = 15
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 256
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip_resizedcrop"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

    vision_only = True

    clm_max_text_len = 256
    clm_do_sample = True
    clm_num_beams = 5


@ex.named_config
def task_finetune_clm_mimic_cxr_language_only():
    exp_name = "task_finetune_clm_mimic_cxr_language_only"
    datasets = ["clm_mimic_cxr"]
    loss_names = _loss_names({"clm": 1})
    batch_size = 128
    max_epoch = 15
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 256
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip_resizedcrop"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

    language_only = True

    clm_max_text_len = 128
    clm_do_sample = True
    clm_num_beams = 5


@ex.named_config
def task_finetune_clm_mimic_cxr_vision_language():
    exp_name = "task_finetune_clm_mimic_cxr_vision_language"
    datasets = ["clm_mimic_cxr"]
    loss_names = _loss_names({"clm": 1})
    batch_size = 128
    max_epoch = 15
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 256
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip_resizedcrop"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

    clm_max_text_len = 128
    clm_do_sample = True
    clm_num_beams = 5


@ex.named_config
def task_finetune_nli_radnli_plus():
    exp_name = "task_finetune_nli_radnli_plus"
    datasets = ["nli_radnli_plus"]
    loss_names = _loss_names({"cls": 1})
    batch_size = 16
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 64
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384

    language_only = True

    label_column_name = "radnli_plus"


@ex.named_config
def task_finetune_cls_chemprot():
    exp_name = "task_finetune_cls_chemprot"
    datasets = ["cls_chemprot"]
    loss_names = _loss_names({"cls": 1})
    batch_size = 16
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    lr_multiplier_head = 10
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 128
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384

    language_only = True

    label_column_name = "chemprot"


@ex.named_config
def task_finetune_cls_melinda_i_meth():
    exp_name = "task_finetune_cls_melinda_i_meth"
    datasets = ["cls_melinda"]
    loss_names = _loss_names({"cls": 1})
    batch_size = 16
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    lr_multiplier_head = 10
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 128
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384

    label_column_name = "i_meth"


@ex.named_config
def task_finetune_cls_melinda_i_meth_label():
    exp_name = "task_finetune_cls_melinda_i_meth_label"
    datasets = ["cls_melinda"]
    loss_names = _loss_names({"cls": 1})
    batch_size = 16
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    lr_multiplier_head = 10
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 128
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384

    label_column_name = "i_meth_label"


@ex.named_config
def task_finetune_cls_melinda_p_meth():
    exp_name = "task_finetune_cls_melinda_p_meth"
    datasets = ["cls_melinda"]
    loss_names = _loss_names({"cls": 1})
    batch_size = 16
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    lr_multiplier_head = 10
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 128
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384

    label_column_name = "p_meth"


@ex.named_config
def task_finetune_cls_melinda_p_meth_label():
    exp_name = "task_finetune_cls_melinda_p_meth_label"
    datasets = ["cls_melinda"]
    loss_names = _loss_names({"cls": 1})
    batch_size = 16
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    lr_multiplier_head = 10
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 128
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384

    label_column_name = "p_meth_label"


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end

# vision encoder
@ex.named_config
def swin32_base224():
    vit = "swin_base_patch4_window7_224_in22k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024


@ex.named_config
def swin32_base384():
    vit = "swin_base_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024


@ex.named_config
def swin32_large384():
    vit = "swin_large_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1536


@ex.named_config
def clip32():
    vit = 'ViT-B/32'
    image_size = 224
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768


@ex.named_config
def clip16():
    vit = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768


# text encoder
@ex.named_config
def text_roberta():
    tokenizer = "roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768


@ex.named_config
def text_roberta_large():
    tokenizer = "roberta-large"
    vocab_size = 50265
    input_text_embed_size = 1024


# random augmentation
@ex.named_config
def imagenet_randaug():
    train_transform_keys = ["imagenet_randaug"]


@ex.named_config
def clip_randaug():
    train_transform_keys = ["clip_randaug"]


@ex.named_config
def clip_resizedcrop():
    train_transform_keys = ["clip_resizedcrop"]
