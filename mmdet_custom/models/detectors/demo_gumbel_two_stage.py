from mmdet.models.builder import DETECTORS
from .gumbel_two_stage import GumbelTwoStageDetector

import sys
sys.path.append("....")
from global_storage.global_storage import __global_storage__

@DETECTORS.register_module()
class DemoGumbelTwoStageDetector(GumbelTwoStageDetector):
    def __init__(self, **kwargs):
        super(DemoGumbelTwoStageDetector, self).__init__(**kwargs)

    #  add selectors
    def extract_feat(self, img, need_loss=False):
        """Extract features from images."""
        x, selectors = self.backbone.forward_demo(img)
        __global_storage__.append(selectors)
        if self.with_neck:
            x = self.neck(x)
        return x