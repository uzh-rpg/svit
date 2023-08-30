from mmcv.parallel import collate
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import replace_ImageToTensor
import numpy as np

def preprocess_wo_normalize(model, imgs):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    #  Remove Normalize
    pipeline = cfg.data.test.pipeline
    for i in range(len(pipeline)):
        if 'transforms' in pipeline[i].keys():
            for transform in pipeline[i]['transforms']:
                if transform['type'] == 'Normalize':
                    transform['mean'] = [0., 0., 0.]
                    transform['std'] = [255., 255., 255.]

    test_pipeline = Compose(pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    return data['img'][0]