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

from .evovit_adapter import EvoViTAdapter
from .adapter_modules import SpatialPriorModule, InteractionBlockForEvo, deform_inputs

import sys
sys.path.append("....")
from global_storage.global_storage import __global_storage__

_logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class DemoEvoViTAdapter(EvoViTAdapter):
    def __init__(self, **kwargs):
        super(DemoEvoViTAdapter, self).__init__(**kwargs)

    def forward(self, x):
        assert self.stage_wise_prune is False, 'Stage-wise prune not implemented in InteractionBlockForEvo'
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        cls_token = self.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token = self.pos_drop(cls_token + self.pos_embed[:, :1])
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        cls_attn = 0

        sele_dict = {}

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            cls_token, x, c, cls_attn, sele_dict_ = layer.forward_demo(cls_token, x, c, indexes, self.blocks[indexes[0]:indexes[-1] + 1],
                                    self.norms[indexes[0]:indexes[-1] + 1], self.vs[indexes[0]:indexes[-1] + 1],
                                    self.qks[indexes[0]:indexes[-1] + 1], self.projs[indexes[0]:indexes[-1] + 1],
                                    deform_inputs1, deform_inputs2, H, W, self.prune_ratio, self.tradeoff, cls_attn)
            sele_dict.update(sele_dict_)
        sele_list = []
        for key in sorted(sele_dict):
            sele_list.append(sele_dict[key])
        __global_storage__.append(sele_list)

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
        return [f1, f2, f3, f4]
