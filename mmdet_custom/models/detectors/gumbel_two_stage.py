import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.models.builder import DETECTORS

@DETECTORS.register_module()
class GumbelTwoStageDetector(TwoStageDetector):
    """ Compared to TwoStageDetector, GumbelTwoStageDetector only adds an additional training loss for the backbone"""

    def __init__(self, ratio_loss_weight=1., **kwargs):
        super(GumbelTwoStageDetector, self).__init__(**kwargs)
        self.ratio_loss_weight = ratio_loss_weight

    def extract_feat(self, img, need_loss=False):
        """Directly extract features from the backbone+neck."""
        out = self.backbone(img, need_loss)
        if need_loss:
            assert isinstance(out, tuple) and len(out) == 2
            x, loss = out
            if self.with_neck:
                x = self.neck(x)
            return x, loss
        else:
            if self.with_neck:
                out = self.neck(out)
            return out

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        x, loss_backbone = self.extract_feat(img, need_loss=True)

        loss_backbone['backbone.ratio_loss'] *= self.ratio_loss_weight
        losses.update(loss_backbone)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
