#  Modification from speed/throughput measurement of evit -- Youwei Liang

import argparse
from datasets import build_dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

parser = argparse.ArgumentParser('speed test script', add_help=False)
parser.add_argument('--data-path', default='../data/imagenet/', type=str,
                        help='dataset path')
parser.add_argument('--input-size', default=224)
parser.add_argument('--data-set', default='IMNET')
parser.add_argument('--use-lmdb', default=False)

all_models = \
"""
deit_small_patch16_224, pretrained/deit_small_patch16_224-cd65a155.pth
svit_s, pretrained/svit.pth
"""

import torch
import time
from timm.models import create_model, load_checkpoint
import selective_vit
import random

NUM_POINTS = 5  # 30

def main():
    args = parser.parse_args()
    BATCH_SIZES =[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    N_TEST = 50
    REPEAT = 2
    WARM_UP = 5
    if not os.path.exists('throughput.txt'):
        with open('throughput.txt', 'w') as f:
            pass
    with open('throughput.txt', 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines if not l.startswith('#') and 't' in l]
    deit_th = {}
    svit_th = {}
    for line in lines:
        mod, bs, th = line.split(' ')
        bs = int(bs)
        th = float(th)
        if mod == 'deit':
            if bs not in deit_th:
                deit_th[bs] = [th]
            else:
                deit_th[bs].append(th)
        else:
            assert mod == 'svit'
            if bs not in svit_th:
                svit_th[bs] = [th]
            else:
                svit_th[bs].append(th)
    times=0
    while True:
        times += 1
        print(f'times={times}')
        BATCH_SIZE = BATCH_SIZES[random.randint(0,len(BATCH_SIZES)-1)]
        progress = (sum([len(value) for value in svit_th.values()]) + sum([len(value) for value in deit_th.values()])) / (2*len(BATCH_SIZES) * NUM_POINTS) * 100
        print(f'progress: {progress}% ')
        if progress >= 100:
            break
        if (BATCH_SIZE in deit_th and BATCH_SIZE in svit_th) and \
                (len(deit_th[BATCH_SIZE])>=NUM_POINTS and (len(svit_th[BATCH_SIZE])>=NUM_POINTS)):
            continue
        print(f'current batch size:{BATCH_SIZE}')

        # ---------------------------------------------------------
        # --------- load images for models ------------
        # ---------------------------------------------------------
        dataset_val, _ = build_dataset(is_train=False, args=args)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE)
        # for i in range(random.randint(0,30)):
        #     input, _ = next(iter(data_loader_val))
        rand_i = random.randint(0, int(1000/BATCH_SIZE))
        input, _ = next(itertools.islice(data_loader_val, rand_i, None))
        input = input.cuda()

        # ---------------------------------------------------------
        # --------- test throughput for models ------------
        # ---------------------------------------------------------
        for line in all_models.split("\n"):
            if line.startswith('_') or line == '':
                continue
            model_name, checkpoint = [x.strip() for x in line.split(',')]
            model = create_model(
                model_name,
                in_chans=3,
                scriptable=False)
            if checkpoint:
                checkpoint = torch.load(checkpoint, map_location='cpu')
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                model.load_state_dict(checkpoint, strict=False)
            model.visualize = False
            model.statistics = False
            model.fast_path = True
            print("created model", model_name)
            model = model.cuda()
            model.eval()

            with torch.no_grad():
                for _ in range(REPEAT):
                    start = time.time()
                    for _ in range(N_TEST):
                        model(input)
                    torch.cuda.synchronize()
                    end = time.time()
                    elapse = end - start
                    speed = BATCH_SIZE * N_TEST / elapse
                    print(f'{model_name}: {speed:.3f}')
            if times <= WARM_UP:
                continue  # Warm up. Speeds not recorded.
            if model_name == 'deit_small_patch16_224':
                mod = 'deit'
                if not BATCH_SIZE in deit_th:
                    deit_th[BATCH_SIZE] = [speed]
                    with open('throughput.txt', 'a') as f:
                        f.write(f'\n {mod} {BATCH_SIZE} {speed}')
                else:
                    if len(deit_th[BATCH_SIZE])< NUM_POINTS:
                        deit_th[BATCH_SIZE].append(speed)
                        with open('throughput.txt', 'a') as f:
                            f.write(f'\n {mod} {BATCH_SIZE} {speed}')
            else:
                assert model_name == 'svit_s'
                mod = 'svit'
                if not BATCH_SIZE in svit_th:
                    svit_th[BATCH_SIZE] = [speed]
                    with open('throughput.txt', 'a') as f:
                        f.write(f'\n {mod} {BATCH_SIZE} {speed}')
                else:
                    if len(svit_th[BATCH_SIZE]) < NUM_POINTS:
                        svit_th[BATCH_SIZE].append(speed)
                        with open('throughput.txt', 'a') as f:
                            f.write(f'\n {mod} {BATCH_SIZE} {speed}')
            del model


    with open('throughput.txt', 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines if not l.startswith('#') and 't' in l]
    deit_th = {}
    svit_th = {}
    for line in lines:
        mod, bs, th = line.split(' ')
        bs = int(bs)
        th = float(th)
        if mod == 'deit':
            if bs not in deit_th:
                deit_th[bs] = [th]
            else:
                deit_th[bs].append(th)
        else:
            assert mod == 'svit'
            if bs not in svit_th:
                svit_th[bs] = [th]
            else:
                svit_th[bs].append(th)

    d1 = [deit_th[1],deit_th[2],deit_th[4],deit_th[8],deit_th[16],deit_th[32],deit_th[64],deit_th[128],deit_th[256],deit_th[512]]
    m1 = np.array([np.mean(x) for x in d1])
    std1 = np.array([np.std(x) for x in d1])

    d2 = [svit_th[1],svit_th[2],svit_th[4],svit_th[8],svit_th[16],svit_th[32],svit_th[64],svit_th[128],svit_th[256],svit_th[512]]
    m2 = np.array([np.mean(x) for x in d2])
    std2 = np.array([np.std(x) for x in d2])

    x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    plt.plot(x,m1, label='DeiT-S')
    plt.plot(x, m2, label='SViT-S')
    plt.legend()
    plt.fill_between(x, m1-std1, m1+std1, color='b', alpha=.1)
    plt.fill_between(x, m2-std2, m2+std2, color='b', alpha=.1)
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput')
    plt.show()


if __name__ == "__main__":
    main()