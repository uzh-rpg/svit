from test_utils import draw_selected_patches
from timm.models.factory import create_model
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import matplotlib.pyplot as plt
from token_stat import selector_record, select_stat
import selective_vit
from change_name_of_state_dict import change_key_name_from_timm_to_pytorch
import argparse

parser = argparse.ArgumentParser(description='Visualize Token Pruning')
parser.add_argument('--image', default='../data/imagenet/val/n01855672/ILSVRC2012_val_00029785.JPEG',
                    help='the path to the input image')
parser.add_argument('--model', default='svit_s')
parser.add_argument('--checkpoint', default='pretrained/svit.pth',
                    help='checkpoint path for the model')
parser.add_argument('--dataset', default='imagenet',
                    help='the preprocessing configuration, choose from {imagenet, cifar100}')
parser.add_argument('--change_key_name', default=False,
                    help='change the state dict key names from timm ViTs to pytorch ViTs')

def visualize_token_selection(model_name='exp0_inner_h4',
                              checkpoint='pretrained/svit.pth',
                              filename='/data/imageNet1k/val/n01558993/ILSVRC2012_val_00029495.JPEG',
                              dataset='imagenet',
                              draw_attn_weights=False,
                              change_key_name=False):
    model = create_model(model_name, pretrained=False, visualize=True)
    state_dict = torch.load(checkpoint, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    if change_key_name:
        state_dict = change_key_name_from_timm_to_pytorch(state_dict)
    model.load_state_dict(state_dict, strict=True)

    # model.set_all_need_attn_weights_to_True(value=None)
    model.eval()


    transform_cifar100 = create_transform((3, 224, 224), is_training=False, use_prefetcher=False, no_aug=False,
                                 scale=None, ratio=None, hflip=0.5, vflip=0.0, color_jitter=0.4,
                                 auto_augment=None, interpolation='bicubic', mean=(0.507, 0.487, 0.441),
                                 std=(0.267, 0.256, 0.276), crop_pct=0.9, tf_preprocessing=False,
                                 re_prob=0.0, re_mode='const', re_count=1, re_num_splits=0, separate=False
                                 )  # Cifar100 preprocessing

    config = resolve_data_config({}, model=model)
    transform_imagenet = create_transform(**config)
    no_normalize_cfg = resolve_data_config({'mean': [0, 0, 0], 'std': [1, 1, 1]}, model=model)
    no_normalize_transform = create_transform(**no_normalize_cfg)
    if dataset == 'imagenet':
        transform = transform_imagenet
    elif dataset == 'cifar100':
        transform = transform_cifar100

    img = Image.open(filename).convert('RGB')
    input = transform(img).unsqueeze(0)
    img_tensor = no_normalize_transform(img).unsqueeze(0)
    plt.imshow(img_tensor[0, ...].permute(1, 2, 0))
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)
    # plt.savefig('demo/origin.png', bbox_inches='tight', pad_inches=0.02)
    plt.show()

    _ = model(input)
    if len(selector_record) == 0:
        selector_record.append(torch.ones(1, 197))
    while len(selector_record) < len(model.blocks):
        selector_record.insert(0, torch.ones_like(selector_record[-1]))
    selectors = torch.cat(selector_record, dim=0).float()
    used_next_layer = torch.zeros_like(selectors)
    used_later_layers = torch.zeros_like(selectors)
    for i in range(len(used_next_layer) - 1):
        used_next_layer[i] = (1 - selectors[i]) * selectors[i + 1]
    for i in range(len(used_later_layers) -1):
        used_later_layers[i] = torch.logical_and(~selectors[i].bool(), selectors[i+1:].sum(0)>=1).float()

    # depth map
    assert len(selectors) == 12
    if selectors.shape[1] == 197:
        depth_map = selectors[:, 1:]
        depth_map = depth_map.sum(0).view(14, 14)
    else:
        depth_map = selectors.sum(0).view(14, 14)
    plt.imshow(depth_map, cmap='hot', vmin=7, vmax=12)
    plt.colorbar()
    plt.show()

    # token selection
    for i in range(len(selectors)):
        # if i == 0 or not torch.equal(selector_record[i], selector_record[i-1]):
        if True:
            draw_selected_patches(img_tensor, selectors[i], used_next_layer[i], used_later_layers[i],
                                  None, name=i) if draw_attn_weights \
                else draw_selected_patches(img_tensor, selectors[i], used_next_layer[i], used_later_layers[i], name=i)


if __name__ == "__main__":
    args = parser.parse_args()
    visualize_token_selection(model_name=args.model,
                              checkpoint=args.checkpoint,
                              filename=args.image,
                              dataset=args.dataset,
                              draw_attn_weights=False,
                              change_key_name=args.change_key_name)
    selector_record.clear()

    all='all'
    for i in range(12):
        print(f'Layer {i + 1} \t {select_stat[i].avg if select_stat[i].avg != 0 else all}')
        select_stat[i].reset()

