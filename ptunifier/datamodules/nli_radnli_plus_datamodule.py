from .base_datamodule import BaseDataModule
from ..datasets import NLIRADNLIPLUSDataset


class NLIRADNLIPLUSDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return NLIRADNLIPLUSDataset

    @property
    def dataset_cls_no_false(self):
        return NLIRADNLIPLUSDataset

    @property
    def dataset_name(self):
        return "nli_radnli_plus"
