import json
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split

from make_arrow import make_arrow_chexpert, make_arrow_pnsa_pneumonia, make_arrow_clm_mimic_cxr


def prepro_vision_mlc_chexpert():
    random.seed(42)

    CHEXPERT_COMPETITION_TASKS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    CHEXPERT_UNCERTAIN_MAPPINGS = {
        "Atelectasis": 1,
        "Cardiomegaly": 0,
        "Consolidation": 0,
        "Edema": 1,
        "Pleural Effusion": 1,
    }

    CHEXPERT_VIEW_COL = "Frontal/Lateral"

    data_root = "data/finetune_vision_data/chexpert/"
    chexpert_train_path = f"{data_root}/CheXpert-v1.0-small/train.csv"
    chexpert_test_path = f"{data_root}/CheXpert-v1.0-small/valid.csv"

    df = pd.read_csv(chexpert_train_path)
    df = df.fillna(0)
    df = df[df["Frontal/Lateral"] == "Frontal"]

    task_dfs = []
    for i, t in enumerate(CHEXPERT_COMPETITION_TASKS):
        index = np.zeros(14)
        index[i] = 1
        df_task = df[
            (df["Atelectasis"] == index[0])
            & (df["Cardiomegaly"] == index[1])
            & (df["Consolidation"] == index[2])
            & (df["Edema"] == index[3])
            & (df["Pleural Effusion"] == index[4])
            & (df["Enlarged Cardiomediastinum"] == index[5])
            & (df["Lung Lesion"] == index[7])
            & (df["Lung Opacity"] == index[8])
            & (df["Pneumonia"] == index[9])
            & (df["Pneumothorax"] == index[10])
            & (df["Pleural Other"] == index[11])
            & (df["Fracture"] == index[12])
            & (df["Support Devices"] == index[13])
            ]
        df_task = df_task.sample(n=200)
        task_dfs.append(df_task)
    df_200 = pd.concat(task_dfs)

    df = pd.read_csv(chexpert_train_path)
    test_df = pd.read_csv(chexpert_test_path)

    df = df[~df["Path"].isin(df_200["Path"])]
    valid_ids = np.random.choice(len(df), size=5000, replace=False)
    valid_df = df.iloc[valid_ids]
    train_df = df.drop(valid_ids, errors="ignore")

    train_df = train_df[train_df[CHEXPERT_VIEW_COL] == "Frontal"]
    valid_df = valid_df[valid_df[CHEXPERT_VIEW_COL] == "Frontal"]
    test_df = test_df[test_df[CHEXPERT_VIEW_COL] == "Frontal"]
    df_200 = df_200[df_200[CHEXPERT_VIEW_COL] == "Frontal"]

    train_df["Path"] = train_df["Path"].map(lambda x: os.path.join(data_root, x))
    valid_df["Path"] = valid_df["Path"].map(lambda x: os.path.join(data_root, x))
    test_df["Path"] = test_df["Path"].map(lambda x: os.path.join(data_root, x))
    df_200["Path"] = df_200["Path"].map(lambda x: os.path.join(data_root, x))

    train_df = train_df.fillna(0)
    valid_df = valid_df.fillna(0)
    test_df = test_df.fillna(0)
    df_200 = df_200.fillna(0)

    uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
    train_df = train_df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
    valid_df = valid_df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
    test_df = test_df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
    df_200 = df_200.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

    train_df_001 = train_df.sample(frac=0.01)
    train_df_01 = train_df.sample(frac=0.1)

    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of valid samples: {len(valid_df)}")
    print(f"Number of test samples: {len(test_df)}")
    print(f"Number of chexpert5x200 samples: {len(df_200)}")

    data = {
        "train": [],
        "train_001": [],
        "train_01": [],
        "val": [],
        "test": [],
    }

    for split, split_data in zip(["train", "train_001", "train_01", "val", "test"],
                                 [train_df, train_df_001, train_df_01, valid_df, test_df]):
        for sample_idx, sample in split_data.iterrows():

            img_path = sample["Path"]
            texts = ["None"]
            label = sample[CHEXPERT_COMPETITION_TASKS].astype(int).tolist()

            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts,
                    "chexpert": label
                })

    make_arrow_chexpert(data, "mlc_chexpert", "data/finetune_vision_arrows/")


def read_dicom_save_png(img_path):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    new_img_path = str(img_path).replace(".dcm", ".png").replace("stage_2_train_images", "stage_2_train_images_png")
    cv2.imwrite(new_img_path, x)
    return new_img_path


def prepro_vision_mlc_pnsa_pneumonia(test_fac=0.15):
    PNEUMONIA_DATA_DIR = Path("data/finetune_vision_data/rsna_pneumonia/")
    PNEUMONIA_ORIGINAL_TRAIN_CSV = PNEUMONIA_DATA_DIR / "stage_2_train_labels.csv"
    PNEUMONIA_TRAIN_CSV = PNEUMONIA_DATA_DIR / "train.csv"
    PNEUMONIA_VALID_CSV = PNEUMONIA_DATA_DIR / "val.csv"
    PNEUMONIA_TEST_CSV = PNEUMONIA_DATA_DIR / "test.csv"
    PNEUMONIA_IMG_DIR = PNEUMONIA_DATA_DIR / "stage_2_train_images"
    PNEUMONIA_TRAIN_PCT = 0.7

    os.makedirs(Path(str(PNEUMONIA_IMG_DIR).replace("stage_2_train_images", "stage_2_train_images_png")), exist_ok=True)

    df = pd.read_csv(PNEUMONIA_ORIGINAL_TRAIN_CSV)

    # create bounding boxes
    def create_bbox(row):
        if row["Target"] == 0:
            return 0
        else:
            x1 = row["x"]
            y1 = row["y"]
            x2 = x1 + row["width"]
            y2 = y1 + row["height"]
            return [x1, y1, x2, y2]

    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

    # aggregate multiple boxes
    df = df[["patientId", "bbox"]]
    df = df.groupby("patientId").agg(list)
    df = df.reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # create labels
    df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

    # no encoded pixels mean healthy
    df["Path"] = df["patientId"].apply(lambda x: PNEUMONIA_IMG_DIR / (x + ".dcm"))

    # split data
    train_df, test_val_df = train_test_split(df, test_size=test_fac * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Target"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Target"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Target"].value_counts())

    train_df_001 = train_df.sample(frac=0.01)
    train_df_01 = train_df.sample(frac=0.1)

    data = {
        "train": [],
        "train_001": [],
        "train_01": [],
        "val": [],
        "test": [],
    }
    for split, split_data in zip(["train", "train_001", "train_01", "val", "test"],
                                 [train_df, train_df_001, train_df_01, valid_df, test_df]):
        for sample_idx, sample in split_data.iterrows():
            img_path = read_dicom_save_png(sample["Path"])
            texts = ["None"]
            label = [sample["Target"]]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts,
                    "pnsa_pneumonia": label
                })

    make_arrow_pnsa_pneumonia(data, "mlc_pnsa_pneumonia", "data/finetune_vision_arrows/")


# def prepro_vision_clm_mimic_cxr(min_length=3):
#     random.seed(42)
#
#     data = {
#         "train": [],
#         "val": [],
#         "test": []
#     }
#     data_root = "data/pretrain_data/mimic_cxr/"
#     image_root = f"{data_root}/files"
#     sectioned_path = f"{data_root}/mimic_cxr_sectioned.csv"
#     metadata_path = f"{data_root}/mimic-cxr-2.0.0-metadata.csv"
#     chexpert_path = f"{data_root}/mimic-cxr-2.0.0-chexpert.csv"
#     split_path = f"{data_root}/mimic-cxr-2.0.0-split.csv"
#
#     sectioned_data = pd.read_csv(sectioned_path)
#     sectioned_data = sectioned_data.set_index("study")
#     metadata = pd.read_csv(metadata_path)
#     chexpert_data = pd.read_csv(chexpert_path)
#     chexpert_data["subject_id_study_id"] = chexpert_data["subject_id"].map(str) + "_" + chexpert_data["study_id"].map(
#         str)
#     chexpert_data = chexpert_data.set_index("subject_id_study_id")
#     chexpert_data = chexpert_data.fillna(0)
#     chexpert_data[chexpert_data == -1] = 0
#     split_data = pd.read_csv(split_path)
#     split_data = split_data.set_index("dicom_id")
#
#     for sample_idx, sample in metadata.iterrows():
#         subject_id = str(sample["subject_id"])
#         study_id = str(sample["study_id"])
#         dicom_id = str(sample["dicom_id"])
#         img_path = os.path.join(image_root,
#                                 "p" + subject_id[:2],
#                                 "p" + subject_id,
#                                 "s" + study_id,
#                                 dicom_id + ".png")
#         split = split_data.loc[dicom_id].split
#         if split == "validate":
#             split = "val"
#         if sample.ViewPosition not in ["AP", "AP"]:
#             continue
#         if (metadata["study_id"] == sample.study_id).sum() > 1:
#             continue
#         if (subject_id + "_" + study_id) not in chexpert_data.index:
#             print("Missing {}".format(subject_id + "_" + study_id))
#             continue
#         if "s" + study_id not in sectioned_data.index:
#             print("Missing {}".format("s" + study_id))
#             continue
#
#         chexpert = chexpert_data.loc[subject_id + "_" + study_id].iloc[2:].astype(int).tolist()
#
#         texts = []
#         impression = []
#         findings = []
#         if not pd.isna(sectioned_data.loc["s" + study_id]["impression"]):
#             texts.append(sectioned_data.loc["s" + study_id]["impression"])
#             impression.append(sectioned_data.loc["s" + study_id]["impression"])
#         if not pd.isna(sectioned_data.loc["s" + study_id]["findings"]):
#             texts.append(sectioned_data.loc["s" + study_id]["findings"])
#             findings.append(sectioned_data.loc["s" + study_id]["findings"])
#
#         texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
#         texts = [text for text in texts if len(text.split()) >= min_length]
#
#         impression = [re.sub(r"\s+", " ", text.strip()) for text in impression]
#         impression = [text for text in impression if len(text.split()) >= min_length]
#
#         findings = [re.sub(r"\s+", " ", text.strip()) for text in findings]
#         findings = [text for text in findings if len(text.split()) >= min_length]
#
#         if len(texts) > 0 and len(impression) == 1 and len(findings) == 1:
#             data[split].append({
#                 "img_path": img_path,
#                 "texts": findings,
#                 "chexpert": chexpert,
#                 "findings": findings,
#                 "impression": impression,
#             })
#     from train_tokenizers import train_tokenizer
#     texts = [sample["findings"][0] for sample in data["train"]] + [sample["impression"][0] for sample in
#                                                                    data["train"]] + \
#             [sample["findings"][0] for sample in data["val"]] + [sample["impression"][0] for sample in data["val"]]
#     train_tokenizer(texts, "data/finetune_vision_arrows/tokenizer-mimic-cxr.json")
#     make_arrow_clm_mimic_cxr(data, "clm_mimic_cxr", "data/finetune_vision_arrows/")


def prepro_vision_clm_mimic_cxr(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }
    data_root = "data/pretrain_data/mimic_cxr/"
    image_root = f"{data_root}/files"
    sectioned_path = f"{data_root}/mimic_cxr_sectioned.csv"
    metadata_path = f"{data_root}/mimic-cxr-2.0.0-metadata.csv"
    chexpert_path = f"{data_root}/mimic-cxr-2.0.0-chexpert.csv"
    split_path = f"{data_root}/mimic-cxr-2.0.0-split.csv"
    presplit_data_path = f"{data_root}/mimic_cxr_presplit.json"

    sectioned_data = pd.read_csv(sectioned_path)
    sectioned_data = sectioned_data.set_index("study")
    metadata = pd.read_csv(metadata_path)
    chexpert_data = pd.read_csv(chexpert_path)
    chexpert_data["subject_id_study_id"] = chexpert_data["subject_id"].map(str) + "_" + chexpert_data["study_id"].map(
        str)
    chexpert_data = chexpert_data.set_index("subject_id_study_id")
    chexpert_data = chexpert_data.fillna(0)
    chexpert_data[chexpert_data == -1] = 0
    split_data = pd.read_csv(split_path)
    split_data = split_data.set_index("dicom_id")
    presplit_data = json.load(open(presplit_data_path))

    for split in ["train", "val", "test"]:
        presplit_split_data = presplit_data[split]
        for sample in presplit_split_data:
            subject_id = str(sample["subject_id"])
            study_id = str(sample["study_id"])
            img_path = os.path.join(image_root, sample["image_path"][0].replace(".jpg", ".png"))
            assert os.path.exists(img_path)

            if (subject_id + "_" + study_id) not in chexpert_data.index:
                print("Missing {}".format(subject_id + "_" + study_id))
                continue
            chexpert = chexpert_data.loc[subject_id + "_" + study_id].iloc[2:].astype(int).tolist()

            findings = [sample["findings"]]
            impression = [sample["impression"]]

            impression = [re.sub(r"\s+", " ", text.strip()) for text in impression]
            impression = [text for text in impression if len(text.split()) >= min_length]

            findings = [re.sub(r"\s+", " ", text.strip()) for text in findings]
            findings = [text for text in findings if len(text.split()) >= min_length]

            if len(impression) == 1 and len(findings) == 1:
                data[split].append({
                    "img_path": img_path,
                    "texts": findings,
                    "chexpert": chexpert,
                    "findings": findings,
                    "impression": impression,
                })

    make_arrow_clm_mimic_cxr(data, "clm_mimic_cxr", "data/finetune_vision_arrows/")


if __name__ == '__main__':
    prepro_vision_mlc_chexpert()
    prepro_vision_mlc_pnsa_pneumonia()
    prepro_vision_clm_mimic_cxr()
