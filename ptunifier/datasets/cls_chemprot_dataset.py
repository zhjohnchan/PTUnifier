import torch

from .base_dataset import BaseDataset


class CLSCHEMPROTDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["cls_chemprot_train"]
        elif split == "val":
            names = ["cls_chemprot_val"]
        elif split == "test":
            names = ["cls_chemprot_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="text_a")
        self.label_column_name = "label"
        self.labels = self.table[self.label_column_name].to_pandas().tolist()
        assert len(self.labels) == len(self.table)

    def __getitem__(self, index):
        return self.get_suite(index)

    def get_suite(self, index):
        ret = dict()
        txt = self.get_text(index)
        ret.update(txt)
        img_index, cap_index = self.index_mapper[index]
        ret["cls_labels"] = self.labels[img_index]
        return ret

    def collate(self, batch, mlm_collator):
        dict_batch = super(CLSCHEMPROTDataset, self).collate(batch, mlm_collator)

        dict_batch["cls_labels"] = torch.tensor([sample["cls_labels"] for sample in batch])
        return dict_batch
