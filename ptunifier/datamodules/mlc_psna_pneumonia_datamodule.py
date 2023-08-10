from .base_datamodule import BaseDataModule
from ..datasets import MLCPNASPNEUMONIADataset


class MLCPNASPNEUMONIADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MLCPNASPNEUMONIADataset

    @property
    def dataset_cls_no_false(self):
        return MLCPNASPNEUMONIADataset

    @property
    def dataset_name(self):
        return "mlc_psna_pneumonia"
