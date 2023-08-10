import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from ptunifier.models.vision_encoders.position_embeddings import get_2d_sincos_pos_embed
from ptunifier.models.vision_encoders.clip_model import Transformer, LayerNorm


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MIMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.patch_size = config["patch_size"]
        self.num_patches = (config["image_size"] // config["patch_size"]) ** 2
        self.decoder_hidden_size = config["mim_decoder_hidden_size"]
        self.decoder_num_layers = config["mim_decoder_num_layers"]
        self.decoder_num_heads = config["mim_decoder_num_heads"]
        self.decoder_num_channels = 3 * config["patch_size"] ** 2

        self.decoder_embed = nn.Linear(self.hidden_size, self.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1,
                                                          self.decoder_hidden_size), requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_hidden_size, int(self.num_patches ** .5), True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.decoder = Transformer(self.decoder_hidden_size, self.decoder_num_layers + 1, self.decoder_num_heads)
        self.decoder_norm = LayerNorm(self.decoder_hidden_size)
        self.decoder_pred = nn.Linear(self.decoder_hidden_size, self.patch_size ** 2 * 3, bias=True)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed.to(x.dtype)

        # apply Transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.decoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


class ITCHead(nn.Module):
    def __init__(self, hidden_size, temp):
        super().__init__()
        self.vision_ln = LayerNorm(hidden_size * 2)
        self.language_ln = LayerNorm(hidden_size * 2)
        self.vision_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.language_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.temp = temp

    def forward(self, image_feats, text_feats, idx=None):
        image_feats = self.vision_proj(self.vision_ln(image_feats))
        text_feats = self.language_proj(self.language_ln(text_feats))

        # normalized features
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        # gather features
        image_feats_all = allgather(image_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feats_all = allgather(text_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())

        # cosine similarity as logits
        logits_per_image = image_feats_all @ text_feats_all.t() / self.temp
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def proj_images(self, image_feats):
        image_feats = self.vision_proj(self.vision_ln(image_feats))
        return image_feats

    def proj_texts(self, text_feats):
        text_feats = self.language_proj(self.language_ln(text_feats))
        return text_feats
