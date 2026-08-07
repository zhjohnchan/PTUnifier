

# PTUnifier
Esta es la implementación de [Towards Unifying Medical Vision-and-Language Pre-training via Soft Prompts](https://arxiv.org/pdf/2302.08958) presentada en ICCV-2023.

## Tabla de Contenidos

- [Requisitos](#requirements)
- [Pre-entrenamiento](#pre-training)
- [Evaluación en Tareas Específicas](#downstream-evaluation)

## Requisitos

Ejecute el siguiente comando para instalar los paquetes requeridos:

```bash
pip install -r requirements.txt
```

## Preparación
Puede (1) preprocesar los datos usted mismo siguiendo las instrucciones; o (2) solicitar directamente nuestros datos preprocesados [aquí (Asegúrese de adjuntar la licencia)](https://drive.google.com/drive/folders/1NjXj4ZXKs72sHbgSLVAboYVWJhrgcNwN?usp=sharing) junto con la licencia **adjunta** de [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) (por ejemplo, **un enlace a la captura de pantalla de la licencia**) (descargue las carpetas `data` y `downloaded`). **Asegúrese de adjuntar la licencia.**

La estructura del proyecto debe ser:
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

## Pre-entrenamiento

### 1. Preparación del Conjunto de Datos

Organice los conjuntos de datos de pre-entrenamiento con la siguiente estructura:

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

### 2. Preprocesamiento

Ejecute el siguiente comando para preprocesar los datos:

```angular2
python prepro/prepro_pretraining_data.py
```

para obtener los siguientes archivos arrow:

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

### 3. Descarga de los pesos inicializados para el pre-entrenamiento

Descargue los pesos inicializados de METER [aquí](https://drive.google.com/file/d/1GucJX0laISLqkTQPfl8zH7x-Gmh3bMLF/view?usp=sharing).

### 4. Pre-entrenamiento

Ahora podemos comenzar a pre-entrenar el modelo ptunifer:

```angular2
bash run_scripts/pretrain_ptunifer.sh
```

## Evaluación en Tareas Específicas

### 1. Preparación del Conjunto de Datos

Organice los conjuntos de datos para el ajuste fino con la siguiente estructura:

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

### 2. Preprocesamiento

Ejecute el siguiente comando para preprocesar los datos:

```angular2
python prepro/prepro_finetuning_data.py
```

para obtener los siguientes archivos arrow:

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

### 3. Ajuste Fino (Fine-Tuning)

Ahora puede comenzar a ajustar finamente el modelo ptunifier:

```
bash run_scripts/finetune_ptunifier.sh
```

Tareas Soportadas:

* Tareas Uni-modales
    * Clasificación Multi-etiqueta en CheXpert
    * Clasificación en RNAS Pneumonia
    * Clasificación en RadNLI
    * Resumenes de Informes de Radiología en MIMIC-CXR
* Tareas Inter-modales
    * Búsqueda Inter-modal en ROCO (Zero-shot)
    * Búsqueda Inter-modal en ROCO (Ajuste Fino)
    * Generación de Informes en MIMIC-CXR
* Tareas Multi-modales
    * Respuesta a Preguntas Visuales en VQA-RAD
    * Respuesta a Preguntas Visuales en SLACK
    * Respuesta a Preguntas Visuales en MedVQA-2019
    *  Resumen Multi-modal de Informes de Radiología en MIMIC-CXR

## Lista de Verificación para Añadir una Nueva Tarea

1. Añada los hiperparámetros en ptunifier/configs.py

2. Añada una nueva cabeza de predicción en ptunifier/modules/prediction_heads.py

3. Añada un nuevo objetivo en ptunifier/modules/objectives.py

4. Añada nuevas métricas y un esquema de registro en ptunifier/modules/ptunifier_utils.py

5. Añada las nuevas cabezas de predicción al optimizador para el multiplicador de lr en ptunifier/modules/ptunifier_utils.py

## Agradecimientos

El código se basa en [ViLT](https://github.com/dandelin/ViLT), [METER](https://github.com/zdou0830/METER)
y [MAE](https://github.com/facebookresearch/mae).
Agradecemos a los autores por su código de código abierto y recomendamos a los usuarios citar sus trabajos cuando sea aplicable.
