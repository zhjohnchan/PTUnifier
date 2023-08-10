import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ptunifier.models import objectives, ptunifier_utils
from ptunifier.models import prediction_heads
from ptunifier.models.language_encoders.bert_model import BertCrossLayer
from ptunifier.models.language_encoders.bert_model_generation import BertGenerationDecoder
from ptunifier.models.ptunifier_utils import init_weights
from ptunifier.models.vision_encoders.clip_model import build_model, adapt_position_encoding


class PTUnifierTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # == Begin: 1. Build Models ==
        bert_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config['tokenizer'],
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])

        resolution_after = config['image_size']
        self.vision_encoder = build_model(config['vit'], resolution_after=resolution_after)
        self.language_encoder = AutoModel.from_pretrained(config['tokenizer'])

        self.multi_modal_language_proj = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.modality_type_embeddings.apply(init_weights)

        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_language_layers.apply(init_weights)

        self.multi_modal_vision_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)
        # == End  : 1. Build Models ==

        # == Begin: 2. Build Pre-Training Heads ==
        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["umlm"] > 0 or config["loss_names"]["clm"] > 0:
            self.mlm_head = prediction_heads.MLMHead(bert_config)
            self.mlm_head.apply(init_weights)
        if config["loss_names"]["mim"] > 0 or config["loss_names"]["umim"] > 0:
            self.mim_head = prediction_heads.MIMHead(config)
            self.mim_head.apply(init_weights)
        if config["loss_names"]["itm"] > 0 or self.hparams.config["loss_names"]["irtr"] > 0:
            self.itm_head = prediction_heads.ITMHead(config["hidden_size"] * 2)
            self.itm_head.apply(init_weights)
        if config["loss_names"]["itc"] > 0:
            self.itc_head = prediction_heads.ITCHead(config["hidden_size"], config["cl_temp"])
            self.itc_head.apply(init_weights)

        self.pseudo_vision_token_pool_size = config["pseudo_vision_token_pool_size"]
        self.pseudo_langauge_token_pool_size = config["pseudo_langauge_token_pool_size"]
        self.num_pseudo_vision_tokens = config["num_pseudo_vision_tokens"]
        self.num_pseudo_langauge_tokens = config["num_pseudo_langauge_tokens"]

        if self.pseudo_vision_token_pool_size > 0:
            self.pseudo_vision_token_pool = torch.nn.Parameter(
                torch.empty((self.pseudo_vision_token_pool_size,
                             self.vision_encoder.visual.width)).normal_(mean=0.0, std=0.02),
                requires_grad=True if "pretrain" in config["exp_name"] else False)
        if self.pseudo_langauge_token_pool_size > 0:
            self.pseudo_language_token_pool = torch.nn.Parameter(
                torch.empty((self.pseudo_langauge_token_pool_size,
                             self.language_encoder.config.hidden_size)).normal_(mean=0.0, std=0.02),
                requires_grad=True if "pretrain" in config["exp_name"] else False)
        # == End  : 2. Build Pre-Training Heads ==

        # == Begin: 3. Load Models ==
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict,
                                                 after=resolution_after,
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
        # == End  : 3. Load Models ==

        # == 4. Build Heads For Downstream Tasks ==
        hs = self.hparams.config["hidden_size"]
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqa_label_size"]
            self.vqa_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_head.apply(init_weights)

        if self.hparams.config["loss_names"]["cls"] > 0:
            ms = self.hparams.config["label_size"][self.hparams.config["label_column_name"]]
            self.cls_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, ms),
            )
            self.cls_head.apply(init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.irtr_head = nn.Linear(hs * 2, 1)
            self.irtr_head.weight.data = self.itm_head.fc.weight.data[1:, :]
            self.irtr_head.bias.data = self.itm_head.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_head.parameters():
                p.requires_grad = False

        if self.hparams.config["loss_names"]["mlc"] > 0:
            ms = self.hparams.config["label_size"][self.hparams.config["label_column_name"]]
            self.mlc_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, ms),
            )
            self.mlc_head.apply(init_weights)

        if self.hparams.config["loss_names"]["clm"] > 0:
            self.clm_tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
            self.clm_proj = nn.Sequential(
                nn.Linear(hs, self.language_encoder.config.hidden_size),
                nn.LayerNorm(self.language_encoder.config.hidden_size)
            )
            self.clm_proj.apply(init_weights)
            self.clm_head = BertGenerationDecoder(config=self.language_encoder.config,
                                                  max_length=self.hparams.config["clm_max_text_len"])
            self.clm_head.apply(init_weights)

            self.clm_head.bert.embeddings.load_state_dict(
                copy.deepcopy(self.language_encoder.embeddings.state_dict()), strict=True
            )
            self.clm_head.bert.encoder.load_state_dict(
                copy.deepcopy(self.language_encoder.encoder.state_dict()), strict=False
            )
            self.clm_head.lm_head.load_state_dict(
                copy.deepcopy(self.mlm_head.state_dict()), strict=True
            )

        ptunifier_utils.set_metrics(self)
        self.current_tasks = list()
        # == End:  4. Build Heads For Downstream Tasks ==

        # == Begin: 5. Load Models For Testing ==
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict, after=resolution_after,
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
        # == End  : 5. Load Models For Testing ==

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=False,
            pseudo_vision=False,
            pseudo_language=False,
    ):
        ret = dict()

        # == Begin: Fetch the inputs ==
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            if pseudo_vision:
                img = None
            else:
                img = batch[img_key][0]
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        device = self.device
        # == End  : Fetch the inputs ==
        # == Begin: Image and Text Embeddings ==
        assert (pseudo_vision & pseudo_language) is False
        if pseudo_vision:
            # text token embeddings
            uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
            if self.num_pseudo_vision_tokens > 0:
                # average pooling for token embeddings
                query_tensors = (uni_modal_text_feats[:, 1:] * text_masks.unsqueeze(-1)[:, 1:]).sum(1) / (
                        text_masks[:, 1:].sum(1, keepdims=True) + 1e-8)
                # find pseudo tokens
                pseudo_tokens = self.find_pseudo_tokens(query_tensors,
                                                        self.pseudo_vision_token_pool,
                                                        self.num_pseudo_vision_tokens)
                # concatenate cls embeds and pseudo tokens
                vision_cls_token = self.vision_encoder.visual.get_pos_encoded_cls_embed()
                vision_cls_token = vision_cls_token.unsqueeze(0).repeat(len(pseudo_tokens), 1, 1)
                uni_modal_image_feats = torch.cat([vision_cls_token, pseudo_tokens], dim=1)
            else:
                # use cls tokens
                vision_cls_token = self.vision_encoder.visual.get_pos_encoded_cls_embed()
                vision_cls_token = vision_cls_token.unsqueeze(0).repeat(len(uni_modal_text_feats), 1, 1)
                uni_modal_image_feats = vision_cls_token

        elif pseudo_language:
            # image token embeddings
            uni_modal_image_feats = self.vision_encoder.forward_patch_embed(img)
            uni_modal_image_feats = self.vision_encoder.forward_pos_embed(uni_modal_image_feats)
            if self.num_pseudo_langauge_tokens > 0:
                # average pooling for token embeddings
                query_tensors = uni_modal_image_feats[:, 1:].mean(1)
                # find pseudo tokens
                pseudo_tokens = self.find_pseudo_tokens(query_tensors,
                                                        self.pseudo_language_token_pool,
                                                        self.num_pseudo_langauge_tokens)
                # concatenate cls embeds, pseudo tokens, and sep embeds
                cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
                cls_id = torch.full((len(pseudo_tokens), 1), cls_id, dtype=torch.long, device=device)
                language_cls_token = self.language_encoder.embeddings(input_ids=cls_id)
                sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
                sep_id = torch.full((len(pseudo_tokens), 1), sep_id, dtype=torch.long, device=device)
                language_sep_token = self.language_encoder.embeddings(input_ids=sep_id)
                uni_modal_text_feats = torch.cat([language_cls_token, pseudo_tokens, language_sep_token], dim=1)
            else:
                # use cls tokens
                cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
                cls_id = torch.full((len(uni_modal_image_feats), 1), cls_id, dtype=torch.long, device=device)
                language_cls_token = self.language_encoder.embeddings(input_ids=cls_id)
                uni_modal_text_feats = language_cls_token

        else:
            uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
            if not mask_image:
                uni_modal_image_feats = self.vision_encoder.forward_patch_embed(img)
                uni_modal_image_feats = self.vision_encoder.forward_pos_embed(uni_modal_image_feats)
        # == End: Image and Text Embeddings ==

        # == Begin: Text Encoding ==
        if pseudo_language:
            text_masks = torch.ones((uni_modal_text_feats.size(0), uni_modal_text_feats.size(1)),
                                    dtype=torch.long, device=device)
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, text_masks.size(), device)

        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)
        # == End  : Text Encoding ==

        # == Begin: Image Encoding ==
        if mask_image:
            # == Begin: Image Masking ==
            # Mask: length -> length * mask_ratio
            # Perform position embedding inside the masking function
            uni_modal_image_feats = self.vision_encoder.forward_patch_embed(img)
            uni_modal_image_feats, mim_masks, mim_ids_restore = self.random_masking(uni_modal_image_feats,
                                                                                    self.hparams.config["mim_prob"])
            uni_modal_image_feats = self.vision_encoder.forward_trans(uni_modal_image_feats)
            ret["mim_masks"] = mim_masks
            ret["mim_ids_restore"] = mim_ids_restore
            # == End  : Image Masking ==
        else:
            uni_modal_image_feats = self.vision_encoder.forward_trans(uni_modal_image_feats)

        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)),
                                 dtype=torch.long, device=device)
        extended_image_masks = self.language_encoder.get_extended_attention_mask(image_masks, image_masks.size(),
                                                                                 device)
        # == End  : Image Encoding ==

        # == Begin: Assign Type Embeddings ==
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )
        # == End  : Assign Type Embeddings ==

        # == Begin: Multi-Modal Fusion ==
        ret["attentions"] = {"text2image_attns": [], "image2text_attns": []} if output_attentions else None
        x, y = uni_modal_text_feats, uni_modal_image_feats
        for layer_idx, (text_layer, image_layer) in enumerate(zip(self.multi_modal_language_layers,
                                                                  self.multi_modal_vision_layers)):
            # == Begin: Fetch the intermediate outputs (different layers to perform MIM) ==
            if mask_image and self.hparams.config["mim_layer"] == layer_idx:
                ret[f"multi_modal_text_feats_{layer_idx}"], ret[f"multi_modal_image_feats_{layer_idx}"] = x, y
            # == End  : Fetch the intermediate outputs (different layers to perform MIM) ==
            # == Begin: Co-Attention ==
            x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
            y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
            x, y = x1[0], y1[0]
            # == End: Co-Attention ==
            # == Begin: For visualization: Return the attention weights ==
            if output_attentions:
                ret["attentions"]["text2image_attns"].append(x1[1:])
                ret["attentions"]["image2text_attns"].append(y1[1:])
            # == End  : For visualization: Return the attention weights ==
        # == End  : Multi-Modal Fusion ==

        # == Begin: == Output Multi-Modal Features ==
        multi_modal_text_feats, multi_modal_image_feats = x, y
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        multi_modal_cls_feats = torch.cat([multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)
        # == End  : == Output Multi-Modal Features ==

        ret.update({
            "images": img,
            "patched_images": self.patchify(img) if img is not None else None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "extended_image_masks": extended_image_masks,
            "extended_text_masks": extended_text_masks,
            "multi_modal_text_feats": multi_modal_text_feats,
            "multi_modal_image_feats": multi_modal_image_feats,
            "multi_modal_cls_feats": multi_modal_cls_feats,
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Pre-Training: Uni-Modal Masked Language Modeling
        if "umlm" in self.current_tasks:
            ret.update(objectives.compute_umlm(self, batch))

        # Pre-Training: Masked Image Modeling
        if "mim" in self.current_tasks:
            ret.update(objectives.compute_mim(self, batch))

        # Pre-Training: Uni-Modal Masked Image Modeling
        if "umim" in self.current_tasks:
            ret.update(objectives.compute_umim(self, batch))

        # Pre-Training: Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Pre-Training: Contrastive Learning
        if "cl" in self.current_tasks:
            ret.update(objectives.compute_itc(self, batch))

        # Fine-Tuning: Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test=test))

        # Fine-Tuning: Image-Text Classification
        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test=test))

        # Fine-Tuning: Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch, test))

        # Fine-Tuning: Image-Text Classification
        if "mlc" in self.current_tasks:
            ret.update(objectives.compute_mlc(self, batch, test=test))

        # Fine-Tuning: Causal Language Modeling
        if "clm" in self.current_tasks:
            ret.update(objectives.compute_clm(self, batch, test=test))

        return ret

    def training_step(self, batch, batch_idx):
        ptunifier_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        ptunifier_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        ptunifier_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        ptunifier_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        ptunifier_utils.set_task(self)
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        ptunifier_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return ptunifier_utils.set_schedule(self)

    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]
        x = x[:, 1:]
        pos_embed = self.vision_encoder.visual.positional_embedding.unsqueeze(0).to(x)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x += pos_embed[:, 1:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is removed
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append cls token
        x_ = x_ + pos_embed[:, :1]
        x_masked = torch.cat((x_, x_masked), dim=1)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        p = self.hparams.config["patch_size"]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.hparams.config["patch_size"]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    @torch.no_grad()
    def find_pseudo_tokens(self, query_tensors, pseudo_token_pool, num_pseudo_tokens):
        bs, dim = query_tensors.shape
        queried_idx = (query_tensors @ pseudo_token_pool.T).topk(num_pseudo_tokens, -1)[1]
        pseudo_tokens = pseudo_token_pool.unsqueeze(0).repeat(bs, 1, 1).gather(
            1, queried_idx.unsqueeze(-1).repeat(1, 1, dim))
        return pseudo_tokens
