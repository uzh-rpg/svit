#///////////// Tesing Images ////////////
test_imgs = ['data/coco/val2017/000000022935.jpg']
__supported_models__ = ['GumbelTwoStageDetector', 'MaskRCNN']


#///////////// Testing Models ///////////
all_models = \
"""
configs/mask_rcnn/vit-adapter-t-3x.py, pretrained/vit-adapter-t-3x.pth
configs/mask_rcnn/svit-adapter-t-0.5x-ftune.py, pretrained/svit-adapter-t-0.5x.pth

configs/mask_rcnn/vit-adapter-s-3x.py, pretrained/vit-adapter-s-3x.pth
configs/mask_rcnn/svit-adapter-s-0.33x-ftune.py, pretrained/svit-adapter-s-0.33x.pth
"""


from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import os.path as osp
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch
import time
from global_storage.global_storage import __global_storage__

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    imgs = test_imgs[0]
    WARM_UP = 100
    N_TEST = 200
    REPEAT = 2

    # build the model from a config file and a checkpoint file
    for line in all_models.split("\n"):
        if line.startswith('_') or line == '':
            continue
        tmp_config, tmp_checkpoint = [x.strip() for x in line.split(',')]
        break
    model = init_detector(tmp_config, None if tmp_checkpoint=='' else tmp_checkpoint, device=args.device)
    assert type(model).__name__ in __supported_models__
    # result = inference_detector(model, args.img)

    # ---------------------------------------------------------
    # ----- prepare the same images ready for all models ------
    # ---------------------------------------------------------
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    datas = []
    for img in imgs:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        assert False, 'CPU inference not supported for testing speed.'


    # ---------------------------------------------------------
    # --------- test throughput for all models ------------
    # ---------------------------------------------------------
    for line in all_models.split("\n"):
        if line.startswith('_') or line == '':
            continue
        model_config, checkpoint = [x.strip() for x in line.split(',')]
        model = init_detector(model_config, checkpoint, device=args.device)
        model.eval()

        print('speed (imgs/s):')
        with torch.no_grad():
            for k in range(WARM_UP):
                model(return_loss=False, rescale=True, **data)
            for k in range(REPEAT):
                start = time.time()
                for i in range(N_TEST):
                    model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                end = time.time()
                elapse = end - start
                speed = N_TEST / elapse
                print(f'{model_config}: {speed:.3f}')
            print('\n')




if __name__ == '__main__':
    args = parse_args()
    assert args.async_test is False
    main(args)
