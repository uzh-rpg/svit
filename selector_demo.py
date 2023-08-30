# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import os.path as osp
import torch
from preprocess_img import preprocess_wo_normalize

from global_storage.global_storage import __global_storage__
from mmdet_custom.models.backbones.base.token_stat import select_stat, rescale_stat

__supported_models__ = ['DemoGumbelTwoStageDetector']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
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
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    assert type(model).__name__ in __supported_models__
    # test a single image
    result = inference_detector(model, args.img)

    mmcv.mkdir_or_exist(args.out)
    out_file = osp.join(args.out, osp.basename(args.img))
    # show the results
    model.show_result(
        args.img,
        result,
        score_thr=args.score_thr,
        show=True,
        bbox_color=args.palette,
        text_color=(200, 200, 200),
        mask_color=args.palette,
        # out_file=out_file
    )

    from draw_functions import draw_selected_patches
    assert len(__global_storage__) == 1
    selectors = __global_storage__[0].squeeze(0)  # (12, h, w)
    import matplotlib.pyplot as plt
    plt.imshow(selectors.sum(0).cpu(), cmap='hot', vmin=7, vmax=12)
    plt.colorbar(fraction=0.030, pad=0.03)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)
    plt.savefig('demo/depth_map', bbox_inches='tight', pad_inches=0.02)
    plt.show()

    used_next_layer = torch.zeros_like(selectors)
    for i in range(len(used_next_layer) - 1):
        used_next_layer[i] = (1 - selectors[i]) * selectors[i + 1]
    used_later_layers = torch.zeros_like(selectors)
    for i in range(len(used_later_layers) -1):
        used_later_layers[i] = torch.logical_and(~selectors[i].bool(), selectors[i+1:].sum(0)>=1).float()

    tensor = preprocess_wo_normalize(model, args.img)
    for i in range(selectors.shape[0]):
        draw_selected_patches(tensor, selectors[i].view(-1),
                              used_next_layer[i].view(-1),
                              used_later_layers[i].view(-1),
                              name=f'demo/{i}.png')

    print('Token Use Rate:')
    for i in range(selectors.shape[0]):
        print(f'{selectors[i].sum() / selectors.shape[1] / selectors.shape[2] * 100:.3f}%')
    print(f'total use rate: {selectors.sum() / (selectors.shape[0] * selectors.shape[1] * selectors.shape[2]):.3f}')


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)