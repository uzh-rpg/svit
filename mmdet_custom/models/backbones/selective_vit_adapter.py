# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from .base.selective_vit import SelectiveVisionTransformer
from .adapter_modules import SpatialPriorModule, InteractionBlockWithSelection, InteractionBlockWithInheritSelection, deform_inputs

_logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class SelectiveViTAdapter(SelectiveVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True,
                 ratio_per_sample=False,
                 *args, **kwargs):

        kwargs.pop('window_attn')
        kwargs.pop('window_size')
        assert kwargs.pop('layer_scale') is False, 'Pytorch Better Transformer does not support layer scale'
        assert kwargs.pop('pretrained') is None, 'use load_from instead of pretrained'
        super().__init__(num_heads=num_heads, *args, **kwargs)
        self.norm = None
        self.head = None



        self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        if not self.inherit_mask:
            self.interactions = nn.Sequential(*[
                InteractionBlockWithSelection(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                                 init_values=init_values, drop_path=self.drop_path_rate,
                                 norm_layer=self.norm_layer, with_cffn=with_cffn,
                                 cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                 extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                                 ratio_per_sample=ratio_per_sample)
                for i in range(len(interaction_indexes))
            ])
        else:
            self.interactions = nn.Sequential(*[
                InteractionBlockWithInheritSelection(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                                                     init_values=init_values, drop_path=self.drop_path_rate,
                                                     norm_layer=self.norm_layer, with_cffn=with_cffn,
                                                     cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                                     extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                                                     ratio_per_sample=ratio_per_sample)
                for i in range(len(interaction_indexes))
            ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x, need_loss=False):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        if self.inherit_mask:
            prev_decision = torch.ones(bs, n, 1, device=x.device)
        ratio_loss = 0.
        num_loss = 0
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            if not self.inherit_mask:
                x, c, layer_ratio_loss, has_loss = layer(x, c, indexes,
                             deform_inputs1, deform_inputs2, H, W,
                             self.blocks, self.selective_modules, self.keep_ratio)
            else:
                x, c, layer_ratio_loss, has_loss, prev_decision = layer(x, c, indexes,
                             deform_inputs1, deform_inputs2, H, W,
                             self.blocks, self.selective_modules, self.keep_ratio, self.sl_loc, prev_decision)
            ratio_loss = ratio_loss + layer_ratio_loss
            num_loss = num_loss + has_loss

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        if need_loss:
            return [f1, f2, f3, f4], {'backbone.ratio_loss': ratio_loss / num_loss}
        else:
            return [f1, f2, f3, f4]

    def forward_demo(self, x, need_loss=False):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Selectors
        selectors = torch.ones(x.shape[0], 12, H, W, device=x.device)  # hard code 12 blocks for DeiT
        sele_dict = {}

        # Interaction
        if self.inherit_mask:
            prev_decision = torch.ones(bs, n, 1, device=x.device)
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            if not self.inherit_mask:
                x, c, sele_dict_ = layer.forward_demo(x, c, indexes,
                             deform_inputs1, deform_inputs2, H, W,
                             self.blocks, self.selective_modules, self.keep_ratio)
            else:
                x, c, sele_dict_, prev_decision = layer.forward_demo(x, c, indexes,
                            deform_inputs1, deform_inputs2, H, W,
                            self.blocks, self.selective_modules, self.keep_ratio, self.sl_loc, prev_decision)
            sele_dict.update(sele_dict_)


        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        selectors = selectors.permute(1, 0, 2, 3)  # (B, 12, H, W) -> (12, B, H, W)
        for i, value in sele_dict.items():
            selectors[i] = value.view(-1, H, W)
        selectors = selectors.permute(1, 0, 2, 3)  # (12, B, H, W) -> (B, 12, H, W)


        return [f1, f2, f3, f4], selectors
