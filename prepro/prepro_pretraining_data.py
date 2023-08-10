import json
import os
import random
import re

import pandas as pd

from make_arrow import make_arrow, make_arrow_mimic_cxr


def prepro_medicat(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/pretrain_data/medicat"
    image_root = f"{data_root}/release/figures/"
    medicat_ann_path = f"{data_root}/release/s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl"

    medicat_samples = [json.loads(sample) for sample in open(medicat_ann_path).read().strip().split("\n")]
    medicat_samples = [sample for sample in medicat_samples if sample["radiology"]]
    indices = list(range(len(medicat_samples)))
    random.shuffle(indices)
    splits = {
        "train": indices[:-2000],
        "val": indices[-2000:-1000],
        "test": indices[-1000:],
    }
    for split, split_indices in splits.items():
        for sample_idx in split_indices:
            sample = medicat_samples[sample_idx]
            img_path = os.path.join(image_root, sample["pdf_hash"] + "_" + sample["fig_uri"])
            texts = []
            if "s2_caption" in sample and len(sample["s2_caption"]) > 0:
                texts.append(sample["s2_caption"])
            if "s2orc_references" in sample and sample["s2orc_references"] is not None and len(
                    sample["s2orc_references"]) > 0:
                texts.extend(sample["s2orc_references"])
            texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
            texts = [text for text in texts if len(text.split()) >= min_length]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts
                })

    make_arrow(data, "medicat", "data/pretrain_arrows/")


def prepro_roco(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }
    roco_data_root = "data/pretrain_data/roco"
    roco_image_root = "data/pretrain_data/roco/{}/radiology/images/"
    medicat_roco_data_root = "data/pretrain_data/medicat"
    medicat_roco_paths = {
        "train": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_train_references.jsonl",
        "val": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_val_references.jsonl",
        "test": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_test_references.jsonl"
    }

    medicat2roco = {}
    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/dlinks.txt", "r") as fp:
            for line in fp:
                str_splits = line.strip().split('\t')
                medicat2roco[str_splits[1].split(' ')[2].split('/')[-1].split('.')[0] + "_" + str_splits[-1]] = \
                    str_splits[0]

    for split, path in medicat_roco_paths.items():
        samples = [json.loads(sample) for sample in open(path).read().strip().split("\n")]
        for sample in samples:
            img_path = os.path.join(roco_image_root.format(split), medicat2roco[sample["roco_image_id"]] + ".jpg")
            texts = []
            if "gorc_references" in sample and sample["gorc_references"] is not None and len(
                    sample["gorc_references"]) > 0:
                texts.extend(sample["gorc_references"])
            texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
            texts = [text for text in texts if len(text.split()) >= min_length]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts
                })

    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/captions.txt", "r") as fp:
            for line in fp:
                str_splits = line.strip().split('\t')
                if len(str_splits) == 2:
                    img_path = os.path.join(roco_image_root.format(split), str_splits[0] + ".jpg")
                    texts = [str_splits[1]]
                    texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
                    texts = [text for text in texts if len(text.split()) >= min_length]
                    if len(texts) > 0:
                        data[split].append({
                            "img_path": img_path,
                            "texts": texts
                        })
    make_arrow(data, "roco", "data/pretrain_arrows/")


def prepro_mimic_cxr(min_length=3):
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
    counter = {}
    for sample_idx, sample in metadata.iterrows():
        subject_id = str(sample["subject_id"])
        study_id = str(sample["study_id"])
        dicom_id = str(sample["dicom_id"])
        img_path = os.path.join(image_root,
                                "p" + subject_id[:2],
                                "p" + subject_id,
                                "s" + study_id,
                                dicom_id + ".png")
        split = split_data.loc[dicom_id].split
        if split == "validate":
            split = "val"
        if sample.ViewPosition not in ["PA", "AP"]:
            continue
        if subject_id + "_" + study_id not in chexpert_data.index:
            print("Missing {}".format(subject_id + "_" + study_id))
            continue
        if "s" + study_id not in sectioned_data.index:
            print("Missing {}".format("s" + study_id))
            continue

        chexpert = chexpert_data.loc[subject_id + "_" + study_id].iloc[2:].astype(int).tolist()

        texts = []
        if not pd.isna(sectioned_data.loc["s" + study_id]["impression"]):
            texts.append(sectioned_data.loc["s" + study_id]["impression"])
        if not pd.isna(sectioned_data.loc["s" + study_id]["findings"]):
            texts.append(sectioned_data.loc["s" + study_id]["findings"])
        texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
        texts = [text for text in texts if len(text.split()) >= min_length]

        if len(texts) > 0:
            data[split].append({
                "img_path": img_path,
                "texts": texts,
                "chexpert": chexpert
            })
            counter[subject_id + "_" + study_id] = 0
    make_arrow_mimic_cxr(data, "mimic_cxr", "data/pretrain_arrows/")


if __name__ == '__main__':
    prepro_medicat()
    prepro_roco()
    prepro_mimic_cxr()
