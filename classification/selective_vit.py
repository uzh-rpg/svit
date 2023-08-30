import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch import Tensor
from timm.models.vision_transformer import Mlp, PatchEmbed, _init_vit_weights, trunc_normal_, named_apply
from TransformerEncoderLayer import TransformerEncoderLayer
import logging
import math
from timm.models.registry import register_model
from functools import partial
from token_stat import select_stat, selector_record, selector_hist, selector_depth, raw_selector_hist
_logger = logging.getLogger(__name__)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def left_align_tokens2(x: Tensor, mask: Tensor):
    """
        x: tensor of shape B, L, D
        mask: boolean tensor of shape B, L
    """

    l_aligned_mask, indexes = torch.sort(mask.int(), dim=1, descending=True, stable=True)
    l_aligned_mask = l_aligned_mask.bool()  # bool --> int (sort) --> bool, because CUDA does not sort boolean tensor
    l_aligned_x = x[torch.arange(x.shape[0], device=x.device).unsqueeze(1), indexes]

    return l_aligned_x, l_aligned_mask


def set_inference(module: nn.Module, name, value: bool):
    if hasattr(module, 'inference'):
        module.inference = value
        # print(f'set {name}.inference to {value} ')


class SelectiveModule(nn.Module):
    def __init__(self, channels, rd_channels, hidden_channels, drop=0.,
                 tau=1.,
                 version=0, inference=False):
        super(SelectiveModule, self).__init__()
        self.inference = inference
        self.norm = nn.LayerNorm(channels)
        self.tau = tau
        self.version = version  # version 0 includes CLS token when selecting, version 1 excludes CLS while selecting
        assert version in (0, 1)
        if self.version == 0 or self.version == 1:
            self.mlp = Mlp(channels, hidden_channels, 2, act_layer=nn.GELU, drop=drop)
        self.gate2 = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        B, L, C = x.shape
        if self.version == 1:
            x = x[:, 1:, :]
        x = self.norm(x)
        x = self.mlp(x)  # shape (B,L,1) or (B,L,2)
        scale = self.gate2(x)  #shape (B,L,1) or (B,L,2)
        if not self.inference:
            selector = F.gumbel_softmax(scale, tau=self.tau, hard=True)[:, :, 0:1]  # shape (B, L, 1)
        else:
            selector = torch.argmin(scale, dim=-1, keepdim=True)
        diff_selector = selector
        if self.version == 1:
            selector = torch.cat((torch.ones(B, 1, 1, device=selector.device), selector), dim=1).bool().squeeze(2)
        else:  # self.version = 0
            selector = selector.bool().squeeze(2)

        return selector, diff_selector


class nnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., init_values=None,
                 drop_path=0., act_layer='gelu', norm_layer=nn.LayerNorm,
                 ):
        super(nnBlock, self).__init__()
        _logger.info('Warning: argument qkv_bias is not used by this model.')
        assert attn_drop == drop, 'attn_drop and drop are the same in nn.TransformerEncoder'
        assert norm_layer == nn.LayerNorm or norm_layer.func == nn.LayerNorm, 'nn.TransformerEncoder only supports LayerNorm'
        assert qkv_bias is True, 'pytorch Transformer uses qkv_bias'
        self.TransformerEncoderLayer = TransformerEncoderLayer(
            dim, num_heads, int(mlp_ratio * dim), dropout=drop, activation=act_layer, batch_first=True, norm_first=True,
            drop_path=drop_path)

    def forward(self, x, src_key_padding_mask=None):
        return self.TransformerEncoderLayer(x, src_key_padding_mask=src_key_padding_mask)


class SelectiveVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, fc_norm=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed,
                 norm_layer=None, act_layer='gelu', select_loc=[7, 8, 9, 10, 11, 12],
                 select_model_id=[0, 0, 0, 1, 1, 1], ratio_loss=False, keep_ratio=None, visualize=False,
                 inherit_mask=False, version=0, last_version=None, statistics=True, ratio_per_sample=False):
        super(SelectiveVisionTransformer, self).__init__()
        self.global_pool = global_pool
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_prefix_tokens = 1 if global_pool == 'token' else 0
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            nnBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    init_values=None, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()

        assert len(select_loc) == len(select_model_id)
        self.num_heads = num_heads
        self.sl_loc = select_loc
        self.sl_model_id = select_model_id
        self.selective_modules = nn.ModuleList([])

        for i in range(len(set(select_model_id))):
            if i == len(set(select_model_id)) - 1 and last_version is not None:
                version = last_version
            self.selective_modules.append(SelectiveModule(
                embed_dim, embed_dim // 4, embed_dim // 4,
                version=version))
        self.ratio_loss = ratio_loss
        if keep_ratio is not None:
            assert len(keep_ratio) == len(set(self.sl_model_id))
            self.keep_ratio = keep_ratio
        self.visualize = visualize
        self.inherit_mask = inherit_mask
        self.normal_path = False
        self.fast_path = False
        self.statistics = statistics
        self.ratio_per_sample = ratio_per_sample
        self.grad_checkpointing = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def _ratio_loss(self, selector: Tensor, ratio=1.):
        if self.ratio_per_sample is True:
            n_tokens = selector.shape[1]
            return ((selector.sum(dim=1)/n_tokens - ratio)**2).mean()
        else:
            return (selector.sum() / (selector.shape[0] * selector.shape[1]) - ratio)**2

    def set_all_inference_to(self, value: bool):
        named_apply(partial(set_inference, value=value), self, name=type(self).__name__)


    def forward_features(self, x):
        if self.ratio_loss:
            ratio_loss = 0.
            num_ratio_loss = 0
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        B, N, C = x.shape
        if self.inherit_mask:
            pre_mask = torch.ones(x.shape[0], x.shape[1], 1, dtype=torch.bool, device=x.device)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError
        else:
            ptr = 0
            sl_loc = self.sl_loc[ptr]
            sl_model_id = self.sl_model_id[ptr]
            for i in range(len(self.blocks)):
                cur_loc = i + 1
                if cur_loc < sl_loc or cur_loc > sl_loc:
                    x = self.blocks[i](x)
                elif cur_loc == sl_loc:
                    select_this_layer = False
                    if ptr == 0 or self.sl_model_id[ptr] > self.sl_model_id[ptr - 1]:
                        select_this_layer = True
                        selector, diff_selector = self.selective_modules[sl_model_id](x)

                        if self.inherit_mask:
                            if self.selective_modules[sl_model_id].version == 1:
                                diff_selector = diff_selector * pre_mask[:, 1:]
                                pre_mask = torch.cat((pre_mask[:, :1], diff_selector), dim=1)
                                selector = torch.cat((torch.ones(B, 1, device=x.device), diff_selector.squeeze(-1)), dim=1).bool()
                            else:
                                assert self.selective_modules[sl_model_id].version == 0
                                diff_selector = diff_selector * pre_mask
                                pre_mask = diff_selector
                                selector = diff_selector.squeeze(-1).bool()
                        if self.ratio_loss:
                            result = self._ratio_loss(diff_selector,
                                                      self.keep_ratio[sl_model_id])
                            ratio_loss += result
                            num_ratio_loss += 1

                    # ----only for visualization statistics. COMMENT THIS LINE during training & inference----
                    if not self.training and self.statistics:
                        select_stat[i].update2(selector.sum().detach(), n=selector.shape[0])  # count num selected tokens
                        if self.visualize:
                            selector_record.append(selector.detach())
                            tmp = selector[:, 1:].detach() if selector.shape[1] == 197 else selector.detach()
                            _ = [selector_hist[i].append(int(x.detach())) for x in tmp.sum(1)]
                            raw_selector_hist[i].append(tmp.float())
                            assert tmp.shape[1] == 196  # (B, 196)
                            tmp = tmp.sum(0)
                            selector_depth[0] = selector_depth[0] + tmp.detach().cpu()
                    # ------------------------------------------------------------------------

                    select_mask = torch.cat((torch.ones(B, 1, 1, device=x.device),
                                             diff_selector), dim=1) \
                        if diff_selector.shape[1] == 196 and self.num_prefix_tokens == 1 \
                        else diff_selector

                    if select_this_layer:
                        x = select_mask * self.blocks[i](x, src_key_padding_mask=~selector) + \
                            (1 - select_mask) * x
                    else:
                        x = select_mask.detach() * self.blocks[i](x, src_key_padding_mask=~selector) + \
                            (1 - select_mask.detach()) * x
                    ptr += 1
                    sl_loc = self.sl_loc[ptr] if ptr < len(self.sl_loc) else sl_loc
                    sl_model_id = self.sl_model_id[ptr] if ptr < len(self.sl_model_id) else sl_model_id
                else:
                    assert False, 'This will not happen'
        x = self.norm(x)
        if self.ratio_loss:
            return x, ratio_loss/num_ratio_loss
        return x


    def forward_features_fast_path(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError
        else:
            ptr = 0
            sl_loc = self.sl_loc[ptr]
            sl_model_id = self.sl_model_id[ptr]
            for i in range(len(self.blocks)):
                cur_loc = i + 1
                if cur_loc < sl_loc:
                    x = self.blocks[i](x)
                else:
                    assert cur_loc == sl_loc
                    if ptr == 0 or self.sl_model_id[ptr] > self.sl_model_id[ptr - 1]:
                        selector, diff_selector = self.selective_modules[sl_model_id](x)

                    # only for visualization statistics. COMMENT THIS LINE during training & inference
                    if not self.training and self.statistics:
                        select_stat[i].update2(selector.sum(), n=selector.shape[0])  # count num selected tokens

                    l_aligned_x, l_aligned_mask = left_align_tokens2(x, selector)
                    nt_x = torch._nested_tensor_from_mask(l_aligned_x, l_aligned_mask, mask_check=False)
                    nt_x = self.blocks[i](nt_x, src_key_padding_mask=None)
                    x.masked_scatter_(selector.unsqueeze(-1), torch.cat(nt_x.unbind(), 0))
                    ptr += 1
                    sl_loc = self.sl_loc[ptr] if ptr < len(self.sl_loc) else sl_loc
                    sl_model_id = self.sl_model_id[ptr] if ptr < len(self.sl_model_id) else sl_model_id
        x = self.norm(x)
        return x


    def forward(self, x):
        if self.fast_path and not self.training:
            x = self.forward_features_fast_path(x)
            x = self.forward_head(x)
            return x
        elif self.ratio_loss:
            x, ratio_loss = self.forward_features(x)
            x = self.forward_head(x)
            return x, ratio_loss
        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
            return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)


@register_model
def svit_s(pretrained=False, **kwargs):
    """ Selective Token Transformer, ViT-Small (Vit-Small/16)
    """
    for key in ['base_keep_rate', 'drop_loc', 'fuse_token']:
        if key in kwargs:
            kwargs.pop(key)
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        select_loc=[4, 5, 6, 7, 8, 9, 10, 11, 12], select_model_id=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        version=1, last_version=0,
                        keep_ratio=[0.7, 0.7, 0.7, 0.49, 0.49, 0.49, 0.343, 0.343, 0.343],
                        **kwargs)
    model = SelectiveVisionTransformer(**model_kwargs)
    return model


