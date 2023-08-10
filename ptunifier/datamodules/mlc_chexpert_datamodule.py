from .base_datamodule import BaseDataModule
from ..datasets import MLCCHEXPERTDataset


class MLCCHEXPERTDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MLCCHEXPERTDataset

    @property
    def dataset_cls_no_false(self):
        return MLCCHEXPERTDataset

    @property
    def dataset_name(self):
        return "mlc_chexpert"
