import torch.nn as nn
import numpy as np
import torch
import math
from einops import rearrange
import matplotlib.pyplot as plt
from typing import List
from timm.models.vision_transformer import VisionTransformer
from copy import deepcopy


class Img2Feat_map(nn.Module):
    def __init__(self, input_size=(3, 224, 224), patch_size=(16, 16)):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        assert self.input_size[1] % self.patch_size[0] == 0 and self.input_size[2] % self.patch_size[1] == 0

    def forward(self, x):
        N, C, H, W = x.shape
        assert (C, H, W) == self.input_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=self.patch_size[0], p2=self.patch_size[1])
        return x


class Feat_map2Img(nn.Module):
    def __init__(self, output_channel=3, is_square=True):
        super(Feat_map2Img, self).__init__()
        self.output_channel = output_channel
        # Fixme: transform rectangular patches to images
        if not is_square:
            raise NotImplementedError

    def forward(self, x):
        N, C, H, W = x.shape
        assert C % self.output_channel == 0
        patch_size = int(math.sqrt(C // self.output_channel))
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size)
        return x


def draw_grid_on_img(tensor: torch.Tensor, patch_size=(16, 16), fig_title=''):
    """
        Args:
            tensor: an image of shape (1, 3, h, w)
        """
    h = tensor.shape[2]
    w = tensor.shape[3]
    tensor = tensor.squeeze(0)
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            tensor[:, i, :] = 1
            tensor[:, :, i] = 1
    plt.imshow(tensor.permute(1, 2, 0))
    plt.title(fig_title)
    plt.show()


def draw_selected_patches(tensor: torch.Tensor, selector, use_next=None, use_later=None, attn_weights=None,
                          smooth=True, normalize=True, patch_size=(16, 16), fig_title='', name='out.png'):
    """
    Args:
        tensor: an image of shape (1, 3, h, w)
        selector: the mask of selected tokens, shape (1, L), L is the number of tokens.
        attn_weights: the attention weight, shape (1, L, L)
    """
    tensor = deepcopy(tensor)
    h = tensor.shape[2]
    w = tensor.shape[3]
    grid_h = h / patch_size[0]
    grid_w = w / patch_size[1]
    cls_is_used = False
    selector = selector.squeeze(0)
    use_next = use_next.squeeze(0)
    use_later = use_later.squeeze(0)
    if selector.shape[0] == 197:
        cls_is_used = selector[0]
        selector = selector[1:]
        use_next = use_next[1:]
        use_later = use_later[1:]
    # for i, is_used in enumerate(selector):
    #     if not is_used:
    #         tensor[0, :, int((i // grid_h) * patch_size[0]):int((i // grid_h + 1) * patch_size[0]),
    #         int((i % grid_w) * patch_size[1]):int((i % grid_w + 1) * patch_size[1])] = 1
    # if cls_is_used:
    #     tensor[0, :, 0:5, 0:5] = 0.5
    plt.imshow(tensor[0, ...].permute(1, 2, 0))
    mask = torch.zeros(4, h, w)
    white = torch.tensor([[1, 1, 1, 0.75]]).unsqueeze(-1).unsqueeze(-1)
    White = torch.tensor([[1, 1, 1, 1]]).unsqueeze(-1).unsqueeze(-1)
    black = torch.tensor([[0, 0, 0, 0.5]]).unsqueeze(-1).unsqueeze(-1)
    Black = torch.tensor([[0, 0, 0, 1]]).unsqueeze(-1).unsqueeze(-1)
    Amber = torch.tensor([[1, 0.75, 0, 1]]).unsqueeze(-1).unsqueeze(-1)
    Aqua = torch.tensor([[0, 1, 1, 1]]).unsqueeze(-1).unsqueeze(-1)
    aqua = torch.tensor([[0, 1, 1, 0.75]]).unsqueeze(-1).unsqueeze(-1)

    for i, is_used in enumerate(selector):
        if not is_used:
            mask[:, int((i // grid_h) * patch_size[0]):int((i // grid_h + 1) * patch_size[0]),
            int((i % grid_w) * patch_size[1]):int((i % grid_w + 1) * patch_size[1])] = White
            # mask[:, int((i // grid_h) * patch_size[0] + 1):int((i // grid_h + 1) * patch_size[0] - 1),
            # int((i % grid_w) * patch_size[1] + 1):int((i % grid_w + 1) * patch_size[1] - 1)] = white

    # for i, is_used in enumerate(use_later):
    #     if is_used:
    #         mask[:, int((i // grid_h) * patch_size[0]):int((i // grid_h + 1) * patch_size[0]),
    #         int((i % grid_w) * patch_size[1]):int((i % grid_w + 1) * patch_size[1])] = aqua
            # mask[:, int((i // grid_h) * patch_size[0]+1):int((i // grid_h + 1) * patch_size[0]-1),
            # int((i % grid_w) * patch_size[1]+1):int((i % grid_w + 1) * patch_size[1]-1)] = white
    #
    # for i, is_used in enumerate(use_next):
    #     if is_used:
    #         mask[:, int((i // grid_h) * patch_size[0]):int((i // grid_h + 1) * patch_size[0]),
    #         int((i % grid_w) * patch_size[1]):int((i % grid_w + 1) * patch_size[1])] = Amber
    #         mask[:, int((i // grid_h) * patch_size[0] + 1):int((i // grid_h + 1) * patch_size[0] - 1),
    #         int((i % grid_w) * patch_size[1] + 1):int((i % grid_w + 1) * patch_size[1] - 1)] = white

    plt.imshow(mask.permute(1, 2, 0))


    if attn_weights is not None:
        attn_weights = deepcopy(attn_weights.detach())
        attn_weights = attn_weights.squeeze(0)[0]
        assert abs(attn_weights.sum() - 1) < 1e-6
        if normalize:
            attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())
        if attn_weights.shape[0] == 197:
            cls_self_attn_weight = attn_weights[0]
            attn_weights = attn_weights[1:]
        heat_map = torch.zeros_like(tensor[0, 0])
        for i, weight in enumerate(attn_weights):
            heat_map[int((i // grid_h) * patch_size[0]):int((i // grid_h + 1) * patch_size[0]),
            int((i % grid_w) * patch_size[1]):int((i % grid_w + 1) * patch_size[1])] = weight
        if smooth:
            def gaussian_kernel(size, sigma):
                size = int(size) // 2
                x, y = np.mgrid[-size:size + 1, -size:size + 1]
                normal = 1 / (2.0 * math.pi * sigma ** 2)
                g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
                return g
            kernel = gaussian_kernel(16, sigma=5)
            kernel = torch.tensor(kernel).unsqueeze(0).unsqueeze(0).float()
            conv = nn.Conv2d(1, 1, kernel_size=16, padding=7, bias=False, stride=1)
            conv.weight.data = kernel
            heat_map = heat_map.unsqueeze(0).unsqueeze(0)
            heat_map = conv(heat_map).squeeze(0).squeeze(0).detach()
        plt.imshow(heat_map, cmap='jet', alpha=0.5)

    plt.title(fig_title)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)
    # plt.savefig(f'demo/{name}.png', bbox_inches='tight', pad_inches=0.02)
    plt.show()


def draw_img(tensor: torch.Tensor, fig_title=''):
    """
    Args:
        tensor: an image of shape (1, 3, h, w)
    """
    tensor = tensor.squeeze(0)
    plt.imshow(tensor.permute(1, 2, 0))
    plt.title(fig_title)
    plt.show()


def draw_feat_map(tensor: torch.Tensor, fig_title=''):
    """
    Visualize the first 36 channels of the feature map in grayscale.
    Args:
        tensor: a feature map of shape (1, channel, h, w)
    """
    tensor = tensor.squeeze(0)  # (channel, h, w)
    n_channels = tensor.shape[0]
    fig, axs = plt.subplots(6, 6, figsize=(5, 5))
    for i, feat_map_2d in enumerate(tensor):  # feat_map_2d: (h, w)
        if i == min(36, n_channels):
            break
        else:
            axs[i // 6, i % 6].imshow(feat_map_2d, cmap='gray')
            axs[i // 6, i % 6].axis('off')
    fig.suptitle(fig_title, fontsize='large', y=0.95)
    plt.show()


def draw_rev_imgs_from_feat_maps(patches: List[torch.Tensor], row=14, col=14, fig_title=''):
    """ convert feature maps back to original-sized images, and visualize them in a grid """
    """ 
    Args:
        feat_maps: a list of feat_maps, each of shape (1, d, h, w), and d is a multiple of 3
        row: number of rows for the plot axis
        col: number of columns for the plot axis
    Note:
        len(feat_maps) = row x col, must be satisfied
    """
    assert len(patches) == row * col
    feat_map2img = Feat_map2Img()
    fig, axs = plt.subplots(row, col, squeeze=False)
    for i in range(row):
        for j in range(col):
            axs[i, j].imshow(feat_map2img(patches[i * col + j])[0, ...].permute(1, 2, 0))
            axs[i, j].axis('off')
    fig.suptitle(fig_title)
    plt.show()


class VisionTransformer_truncate(VisionTransformer):
    ''' output intermediate feature maps '''

    def __init__(self, **kwargs):
        super(VisionTransformer_truncate, self).__init__(**kwargs)

    def forward_truncate(self, x, num_blocks=0, norm=False):
        x = self.patch_embed(x)
        if num_blocks == 0:
            return x
        else:
            x = self._pos_embed(x)
        for i in range(num_blocks):
            x = self.blocks[i](x)
        if norm:
            x = self.norm(x)
        return x



def draw_performance_figure(data_list: List, y_axis_name='Top-1 Accuracy (%)', x_axis_name='GMACs'):
    """
    Args:
        data_list: a list of data
               useage example:
               data_list=[data_0, data_1 ... data_n]
               data_0 = ('type_name_0', ((x1,y1,size1), (x2,y2,size2), ... (xk,yk,sizek)))
               ...
               data_n = ('type_name_n', ((x1,y1,size1), (x2,y2,size2), ... (xt,yt,sizet)))
        y_axis_name: name of y axis
        x_axis_name: name of x axis
    """
    color_mapping = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    for c_idx, data in enumerate(data_list):
        c = color_mapping[c_idx]
        model_type, data_pts = data
        x = [pt[0] for pt in data_pts]
        y = [pt[1] for pt in data_pts]
        size = [pt[2] * 50 for pt in data_pts]
        c_seq = [c] * len(data_pts)
        plt.scatter(x, y, s=size, c=c_seq, marker='o', alpha=0.8, label=model_type)
        plt.plot(x, y, c=c, alpha=0.3)
    plt.legend()
    plt.grid(True)
    plt.show()
