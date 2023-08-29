# PTUnifier
This is the implementation of [Towards Unifying Medical Vision-and-Language Pre-training via Soft Prompts](https://arxiv.org/pdf/2302.08958) at ICCV-2023.

## Table of Contents

- [Requirements](#requirements)
- [Pre-training](#pre-training)
- [Downstream Evaluation](#downstream-evaluation)

## Requirements

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Preparation
You can either (1) preprocess the data by yourself following the instruction; or (2) directly apply for our preprocessed data [here](https://drive.google.com/drive/folders/1NjXj4ZXKs72sHbgSLVAboYVWJhrgcNwN?usp=sharing) with the **attached** [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) license (e.g., **a link to the screenshot of the license**) (download folder `data` and `downloaded` folder). The project structure should be:

```
root:[.]
+--ptunifier
| +--datasets
| +--datamodules
| +--metrics
| +--models
| +--config.py
| +--__init__.py
+--prepro
| +--glossary.py
| +--make_arrow.py
| +--prepro_finetuning_language_data.py
| +--prepro_finetuning_data.py
| +--prepro_finetuning_vision_data.py
| +--prepro_pretraining_data.py
+--data
| +--pretrain_arrows
| +--finetune_arrows
| +--finetune_vision_arrows
| +--finetune_language_arrows
+--downloaded
| +--roberta-base
| +--biomed_roberta_base
| +--scorers
| +--meter.ckpt
| +--ptunifier.ckpt
+--run_scripts
| +--pretrain.sh
| +--finetune.sh
+--zero_shot_evaluations
| +--zero_shot_classification_chexpert_5x200.py
+--tools
| +--visualize_datasets.py
| +--convert_meter_weights.py
+--requirements.txt
+--README.md
+--main.py
```

## Pre-training

### 1. Dataset Preparation

Please organize the pre-training datasets as the following structure:

```
root:[data]
+--pretrain_data
| +--roco
| | +--val
| | +--test
| | +--train
| +--mimic_cxr
| | +--files
| | +--mimic-cxr-2.0.0-split.csv
| | +--mimic-cxr-2.0.0-metadata.csv
| | +--mimic-cxr-2.0.0-chexpert.csv
| | +--mimic_cxr_sectioned.csv
| +--medicat
| | +--release
| | +--net
```

### 2. Pre-processing

Run the following command to pre-process the data:

```angular2
python prepro/prepro_pretraining_data.py
```

to get the following arrow files:

```angular2
root:[data]
+--pretrain_arrows
| +--medicat_train.arrow
| +--medicat_val.arrow
| +--medicat_test.arrow
| +--roco_train.arrow
| +--roco_val.arrow
| +--roco_test.arrow
| +--mimic_cxr_train.arrow
| +--mimic_cxr_val.arrow
| +--mimic_cxr_test.arrow
```

### 3. Download the initialized weights for pre-training

Download the initialized meter weights [here](https://drive.google.com/file/d/1GucJX0laISLqkTQPfl8zH7x-Gmh3bMLF/view?usp=sharing).

### 4. Pre-training

Now we can start to pre-train the ptunifer model:

```angular2
bash run_scripts/pretrain_ptunifer.sh
```

## Downstream Evaluation

### 1. Dataset Preparation

Please organize the fine-tuning datasets as the following structure:

```
root:[data]
+--finetune_data
| +--melinda
| | +--train.csv
| | +--dev.csv
| | +--test.csv
| | +--melinda_images
| +--slack
| | +--train.json
| | +--validate.json
| | +--test.json
| | +--imgs
| +--vqa_rad
| | +--trainset.json
| | +--valset.json
| | +--testset.json
| | +--images
| +--medvqa_2019
| | +--val
| | +--test
| | +--train
```

```
+--finetune_vision_data
| +--chexpert
| | +--CheXpert-v1.0-small
| +--rsna_pneumonia
| | +--stage_2_test_images
| | +--stage_2_train_labels.csv
| | +--stage_2_train_images
```

```
+--finetune_language_data
| +--mednli
| | +--mli_train_v1.jsonl
| | +--mli_test_v1.jsonl
| | +--mli_dev_v1.jsonl
| +--radnli
| | +--radnli_pseudo-train.jsonl
| | +--radnli_test_v1.jsonl
| | +--radnli_dev_v1.jsonl
```

### 2. Pre-processing

Run the following command to pre-process the data:

```angular2
python prepro/prepro_finetuning_data.py
```

to get the following arrow files:

```
root:[data]
+--finetune_arrows
| +--vqa_vqa_rad_train.arrow
| +--vqa_vqa_rad_val.arrow
| +--vqa_vqa_rad_test.arrow
| +--vqa_slack_train.arrow
| +--vqa_slack_test.arrow
| +--vqa_slack_val.arrow
| +--vqa_medvqa_2019_train.arrow
| +--vqa_medvqa_2019_val.arrow
| +--vqa_medvqa_2019_test.arrow
| +--irtr_roco_train.arrow
| +--irtr_roco_val.arrow
| +--irtr_roco_test.arrow
```

```
+--finetune_vision_arrows
| +--mlc_chexpert_train_001.arrow
| +--mlc_chexpert_train_01.arrow
| +--mlc_chexpert_train.arrow
| +--mlc_chexpert_val.arrow
| +--mlc_chexpert_test.arrow
| +--mlc_pnsa_pneumonia_train_001.arrow
| +--mlc_pnsa_pneumonia_train_01.arrow
| +--mlc_pnsa_pneumonia_train.arrow
| +--mlc_pnsa_pneumonia_val.arrow
| +--mlc_pnsa_pneumonia_test.arrow
| +--clm_mimic_cxr_train.arrow
| +--clm_mimic_cxr_val.arrow
| +--clm_mimic_cxr_test.arrow
```

```
+--finetune_language_arrows
| +--nli_radnli_plus_train.arrow
| +--nli_radnli_plus_val.arrow
| +--nli_radnli_plus_test.arrow
```

### 3. Fine-Tuning

Now you can start to fine-tune the ptunifier model:

```
bash run_scripts/finetune_ptunifier.sh
```

Supported Tasks:

* Uni-modal Tasks
    * Multi-label Classification on CheXpert
    * Classification on RNAS Pneumonia
    * Classification on RadNLI
    * Radiology Report Summarization on MIMIC-CXR
* Cross-modal Tasks
    * Cross-modal Retrieval on ROCO (Zero-shot)
    * Cross-modal Retrieval on ROCO (Fine-tuned)
    * Report Report Generation on MIMIC-CXR
* Multi-modal Tasks
    * Visual Question Answering on VQA-RAD
    * Visual Question Answering on SLACK
    * Visual Question Answering on MedVQA-2019
    * Multi-modal Radiology Report Summarization on MIMIC-CXR

## Checklist for Adding a New Task

1. Add the hyper-parameters in ptunifier/configs.py

2. Add a new head in ptunifier/modules/prediction_heads.py

3. Add a new objective in ptunifier/modules/objectives.py

4. Add new metrics and logging scheme in ptunifier/modules/ptunifier_utils.py

5. Add the new prediction heads to the optimizer for lr multiplier in ptunifier/modules/ptunifier_utils.py

## Acknowledgement

The code is based on [ViLT](https://github.com/dandelin/ViLT), [METER](https://github.com/zdou0830/METER)
and [MAE](https://github.com/facebookresearch/mae).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.
