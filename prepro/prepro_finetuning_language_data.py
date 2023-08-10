import csv
import json
import random

from make_arrow import make_arrow_text_classification, make_arrow_text_nli


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


def prepro_language_cls_chemprot():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_language_data/ChemProt/"
    chem_pattern = '@CHEMICAL$'
    gene_pattern = '@GENE$'
    chem_gene_pattern = "@CHEMICAL-GENE$"
    label_set = {'CPR:3': 0, 'CPR:4': 1, 'CPR:5': 2, 'CPR:6': 3, 'CPR:9': 4, 'false': 5}

    for split in ["train", "dev", "test"]:
        samples = read_tsv(f"{data_root}/{split}.tsv")
        if split == "dev":
            split = "val"  # change split name from dev to val
        for i, line in enumerate(samples):
            if i == 0:
                continue  # skip tsv header
            guid = "%s-%s-%s" % (str(i), split, line[0])
            if True:  # line[2] != 'false':
                text_a = line[1]
                text_a = text_a.replace('@CHEMICAL$', chem_pattern).replace(
                    '@GENE$', gene_pattern).replace(
                    '@CHEM-GENE$', chem_gene_pattern)
                label = line[2]
            data[split].append({
                "guid": guid,
                "text_a": [text_a],
                "label": label_set[label]
            })
    make_arrow_text_classification(data, "cls_chemprot", "data/finetune_language_arrows/")


def prepro_language_nli_radnli_plus():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_language_data/radnli/"
    radnli_train_path = f"{data_root}/radnli/radnli_pseudo-train.jsonl"
    radnli_val_path = f"{data_root}/radnli/radnli_dev_v1.jsonl"
    radnli_test_path = f"{data_root}/radnli/radnli_test_v1.jsonl"

    mednli_train_path = f"{data_root}/mednli/mli_train_v1.jsonl"

    label_set = {'contradiction': 0, "entailment": 1, "neutral": 2}

    pair_counter = 0
    for split, split_path in zip(["train", "val", "test"], [radnli_train_path, radnli_val_path, radnli_test_path]):
        split_data = open(split_path).read().strip().split("\n")
        for sample in split_data:
            sample = json.loads(sample)
            guid = f"{pair_counter}-radnli-{split}"
            text_a = sample["sentence1"].strip()
            text_b = sample["sentence2"].strip()
            if text_a[-1] != ".":
                text_a = text_a + "."
            if text_b[-1] != ".":
                text_b = text_b + "."
            text = f"<s> {text_a} </s> </s> {text_b} </s>"
            label = sample["gold_label"]

            data[split].append({
                "guid": guid,
                "text": [text],
                "text_a": [text_a],
                "text_b": [text_b],
                "label": label_set[label]
            })

            pair_counter = pair_counter + 1

    mednli_data = open(mednli_train_path).read().strip().split("\n")
    for sample in mednli_data:
        sample = json.loads(sample)

        guid = f"{pair_counter}-mednli-train"
        text_a = sample["sentence1"].strip()
        text_b = sample["sentence2"].strip()
        if text_a[-1] != ".":
            text_a = text_a + "."
        if text_b[-1] != ".":
            text_b = text_b + "."
        text = f"<s> {text_a} </s> </s> {text_b} </s>"
        label = sample["gold_label"]

        data["train"].append({
            "guid": guid,
            "text_a": [text_a],
            "text_b": [text_b],
            "text": [text],
            "label": label_set[label]
        })

        pair_counter = pair_counter + 1

    make_arrow_text_nli(data, "nli_radnli_plus", "data/finetune_language_arrows/")


if __name__ == '__main__':
    prepro_language_cls_chemprot()
    prepro_language_nli_radnli_plus()
