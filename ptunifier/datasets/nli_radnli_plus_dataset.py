import torch

from .base_dataset import BaseDataset


class NLIRADNLIPLUSDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["nli_radnli_plus_train"]
        elif split == "val":
            names = ["nli_radnli_plus_val"]
        elif split == "test":
            names = ["nli_radnli_plus_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="text")
        self.label_column_name = "label"
        self.labels = self.table[self.label_column_name].to_pandas().tolist()
        self.all_text_a = self.table["text_a"].to_pandas().tolist()
        self.all_text_b = self.table["text_b"].to_pandas().tolist()
        assert len(self.labels) == len(self.table)

    def __getitem__(self, index):
        return self.get_suite(index)

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        # text = self.all_texts[index][caption_index]
        text_a = self.all_text_a[index][caption_index]
        text_b = self.all_text_b[index][caption_index]

        encoding = self.tokenizer(
            text_a,
            text_b,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )

        return {
            "text": ((text_a, text_b), encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_suite(self, index):
        ret = dict()
        txt = self.get_text(index)
        ret.update(txt)
        img_index, cap_index = self.index_mapper[index]
        ret["cls_labels"] = self.labels[img_index]
        return ret

    def collate(self, batch, mlm_collator):
        dict_batch = super(NLIRADNLIPLUSDataset, self).collate(batch, mlm_collator)

        dict_batch["cls_labels"] = torch.tensor([sample["cls_labels"] for sample in batch])
        return dict_batch
