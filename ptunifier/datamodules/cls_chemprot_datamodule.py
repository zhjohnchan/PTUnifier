from .base_datamodule import BaseDataModule
from ..datasets import CLSCHEMPROTDataset


class CLSCHEMPROTDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CLSCHEMPROTDataset

    @property
    def dataset_cls_no_false(self):
        return CLSCHEMPROTDataset

    @property
    def dataset_name(self):
        return "cls_chemprot"
