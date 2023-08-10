from transformers import BertTokenizerFast

from .base_dataset import BaseDataset


def get_mimic_cxr_tokenizer(path):
    return BertTokenizerFast(vocab_file="", tokenizer_file=path)


class CLMMIMICCXRDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["clm_mimic_cxr_train"]
        elif split == "val":
            names = ["clm_mimic_cxr_val"]
        elif split == "test":
            names = ["clm_mimic_cxr_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        self.all_findings = self.table["findings"].to_pandas().tolist()
        self.all_impression = self.table["impression"].to_pandas().tolist()

    def __getitem__(self, index):
        return self.get_suite(index)

    def get_suite(self, index):
        ret = super(CLMMIMICCXRDataset, self).get_suite(index)
        img_index, cap_index = self.index_mapper[index]
        ret["findings"] = self.all_findings[img_index][cap_index]
        ret["impression"] = self.all_impression[img_index][cap_index]

        return ret

    def collate(self, batch, mlm_collator):
        dict_batch = super(CLMMIMICCXRDataset, self).collate(batch, mlm_collator)

        dict_batch["findings"] = [sample["findings"].lower() for sample in batch]
        dict_batch["impression"] = [sample["impression"].lower() for sample in batch]

        return dict_batch
