from .clm_mimic_cxr_datamodule import CLMMIMICCXRDataModule
from .cls_chemprot_datamodule import CLSCHEMPROTDataModule
from .cls_melinda_datamodule import CLSMELINDADataModule
from .irtr_roco_datamodule import IRTRROCODataModule
from .mlc_chexpert_datamodule import MLCCHEXPERTDataModule
from .mlc_psna_pneumonia_datamodule import MLCPNASPNEUMONIADataModule
from .nli_radnli_plus_datamodule import NLIRADNLIPLUSDataModule
from .pretraining_medicat_datamodule import MedicatDataModule
from .pretraining_mimic_cxr_datamodule import MIMICCXRDataModule
from .pretraining_roco_datamodule import ROCODataModule
from .vqa_medvqa_2019_datamodule import VQAMEDVQA2019DataModule
from .vqa_slack_datamodule import VQASLACKDataModule
from .vqa_vqa_rad_datamodule import VQAVQARADDataModule

_datamodules = {
    "medicat": MedicatDataModule,
    "roco": ROCODataModule,
    "mimic_cxr": MIMICCXRDataModule,
    "vqa_vqa_rad": VQAVQARADDataModule,
    "vqa_slack": VQASLACKDataModule,
    "vqa_medvqa_2019": VQAMEDVQA2019DataModule,
    "cls_melinda": CLSMELINDADataModule,
    "irtr_roco": IRTRROCODataModule,
    "mlc_chexpert": MLCCHEXPERTDataModule,
    "mlc_psna_pneumonia": MLCPNASPNEUMONIADataModule,
    "clm_mimic_cxr": CLMMIMICCXRDataModule,
    "cls_chemprot": CLSCHEMPROTDataModule,
    "nli_radnli_plus": NLIRADNLIPLUSDataModule,
}
