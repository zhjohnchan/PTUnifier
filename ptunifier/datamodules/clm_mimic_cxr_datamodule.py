from .base_datamodule import BaseDataModule
from ..datasets import CLMMIMICCXRDataset


class CLMMIMICCXRDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CLMMIMICCXRDataset

    @property
    def dataset_cls_no_false(self):
        return CLMMIMICCXRDataset

    @property
    def dataset_name(self):
        return "clm_mimic_cxr"
