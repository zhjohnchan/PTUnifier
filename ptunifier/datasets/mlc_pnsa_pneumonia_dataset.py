import torch

from .base_dataset import BaseDataset


class MLCPNASPNEUMONIADataset(BaseDataset):
    def __init__(self, *args, split="", data_frac=1., **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train" and data_frac == 1.:
            names = ["mlc_pnsa_pneumonia_train"]
        elif split == "train" and data_frac == .1:
            names = ["mlc_pnsa_pneumonia_train_01"]
        elif split == "train" and data_frac == .01:
            names = ["mlc_pnsa_pneumonia_train_001"]
        elif split == "val":
            names = ["mlc_pnsa_pneumonia_val"]
        elif split == "test":
            names = ["mlc_pnsa_pneumonia_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, data_frac=data_frac, names=names, text_column_name="caption")
        self.label_column_name = self.label_column_name
        self.labels = self.table[self.label_column_name].to_pandas().tolist()
        assert len(self.labels) == len(self.table)

    def __getitem__(self, index):
        return self.get_suite(index)

    def get_suite(self, index):
        ret = super(MLCPNASPNEUMONIADataset, self).get_suite(index)
        img_index, cap_index = self.index_mapper[index]
        ret["mlc_labels"] = self.labels[img_index]
        return ret

    def collate(self, batch, mlm_collator):
        dict_batch = super(MLCPNASPNEUMONIADataset, self).collate(batch, mlm_collator)

        dict_batch["mlc_labels"] = torch.tensor([sample["mlc_labels"] for sample in batch], dtype=torch.long)
        return dict_batch
