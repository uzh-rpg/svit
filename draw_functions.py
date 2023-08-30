import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy

def draw_selected_patches(tensor: torch.Tensor, selector, use_next=None, use_later=None, attn_weights=None, smooth=True, normalize=True, patch_size=(16, 16), fig_title='', name='debug.png'):
    """
    Args:
        tensor: an image of shape (1, 3, h, w)
        selector: the mask of selected tokens, shape (1, L), L=h*w/16^2 is the number of tokens.
        attn_weights: the attention weight, shape (1, L, L)
    """
    tensor = deepcopy(tensor)
    h = tensor.shape[2]
    w = tensor.shape[3]
    grid_h = h / patch_size[0]
    grid_w = w / patch_size[1]
    cls_is_used = False
    selector = selector.squeeze(0)
    if selector.shape[0] == grid_h * grid_w + 1:
        cls_is_used = selector[0]
        selector = selector[1:]
        assert False
    plt.imshow(tensor[0, ...].permute(1, 2, 0))
    mask = torch.zeros(4, h, w)
    white = torch.tensor([[1, 1, 1, 0.75]]).unsqueeze(-1).unsqueeze(-1)
    White = torch.tensor([[1, 1, 1, 1]]).unsqueeze(-1).unsqueeze(-1)
    black = torch.tensor([[0, 0, 0, 0.5]]).unsqueeze(-1).unsqueeze(-1)
    Black = torch.tensor([[0, 0, 0, 1]]).unsqueeze(-1).unsqueeze(-1)
    Amber = torch.tensor([[1, 0.75, 0, 1]]).unsqueeze(-1).unsqueeze(-1)
    Aqua = torch.tensor([[0, 1, 1, 1]]).unsqueeze(-1).unsqueeze(-1)
    aqua = torch.tensor([[0, 1, 1, 0.75]]).unsqueeze(-1).unsqueeze(-1)
    lightblue = torch.tensor([[0.64, 0.92, 0.99, 0.75]]).unsqueeze(-1).unsqueeze(-1)
    Lightblue = torch.tensor([[0.64, 0.92, 0.99, 1]]).unsqueeze(-1).unsqueeze(-1)
    for i, is_used in enumerate(selector):
        if not is_used:
            mask[:, int((i // grid_w) * patch_size[0]):int((i // grid_w + 1) * patch_size[0]),
            int((i % grid_w) * patch_size[1]):int((i % grid_w + 1) * patch_size[1])] = White
            # mask[:, int((i // grid_w) * patch_size[0] + 1):int((i // grid_w + 1) * patch_size[0] - 1),
            # int((i % grid_w) * patch_size[1] + 1):int((i % grid_w + 1) * patch_size[1] - 1)] = black

    for i, is_used in enumerate(use_later):
        if is_used:
            mask[:, int((i // grid_w) * patch_size[0]):int((i // grid_w + 1) * patch_size[0]),
            int((i % grid_w) * patch_size[1]):int((i % grid_w + 1) * patch_size[1])] = Aqua
            # mask[:, int((i // grid_w) * patch_size[0]+1):int((i // grid_w + 1) * patch_size[0]-1),
            # int((i % grid_w) * patch_size[1]+1):int((i % grid_w + 1) * patch_size[1]-1)] = white

    # for i, is_used in enumerate(use_next):
    #     if is_used:
    #         mask[:, int((i // grid_w) * patch_size[0]):int((i // grid_w + 1) * patch_size[0]),
    #         int((i % grid_w) * patch_size[1]):int((i % grid_w + 1) * patch_size[1])] = Amber
    #         mask[:, int((i // grid_w) * patch_size[0] + 1):int((i // grid_w + 1) * patch_size[0] - 1),
    #         int((i % grid_w) * patch_size[1] + 1):int((i % grid_w + 1) * patch_size[1] - 1)] = white

    if cls_is_used:
        tensor[0, :, 0:5, 0:5] = 0.5
        assert False, 'deprecated cls plotting'
    plt.imshow(mask.permute(1, 2, 0))

    if attn_weights is not None:
        attn_weights = deepcopy(attn_weights.detach())
        attn_weights = attn_weights.squeeze(0)[0]
        assert abs(attn_weights.sum() - 1) < 1e-6
        if normalize:
            attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())
        if attn_weights.shape[0] == grid_h * grid_w + 1:
            cls_self_attn_weight = attn_weights[0]
            attn_weights = attn_weights[1:]
        heat_map = torch.zeros_like(tensor[0, 0])
        for i, weight in enumerate(attn_weights):
            heat_map[int((i // grid_w) * patch_size[0]):int((i // grid_w + 1) * patch_size[0]),
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
    plt.savefig(name, bbox_inches='tight', pad_inches=0.02)

    plt.show()


def draw_patches_from_idx(tensor: torch.Tensor, idx, patch_size=(16, 16)):
    """
        Args:
            tensor: an image of shape (1, 3, h, w)
            idx: the index of selected tokens, (1, k),
                 k is the number of selected tokens, typically less than the total number of tokens
        """
    tensor = deepcopy(tensor)
    h = tensor.shape[2]
    w = tensor.shape[3]
    grid_h = h / patch_size[0]
    grid_w = w / patch_size[1]
    idx = idx.squeeze(0)
    assert idx.ndim == 1
    plt.imshow(tensor[0, ...].permute(1, 2, 0))

    mask = torch.zeros(4, h, w)
    white = torch.tensor([[1, 1, 1, 0.7]]).unsqueeze(-1).unsqueeze(-1)
    black = torch.tensor([[0, 0, 0, 0.5]]).unsqueeze(-1).unsqueeze(-1)
    Black = torch.tensor([[0, 0, 0, 1]]).unsqueeze(-1).unsqueeze(-1)
    for i in range(int(grid_h * grid_w)):
        if i not in idx:
            mask[:, int((i // grid_w) * patch_size[0]):int((i // grid_w + 1) * patch_size[0]),
            int((i % grid_w) * patch_size[1]):int((i % grid_w + 1) * patch_size[1])] = white

            # mask[:, int((i // grid_w) * patch_size[0]+1):int((i // grid_w + 1) * patch_size[0]-1),
            # int((i % grid_w) * patch_size[1]+1):int((i % grid_w + 1) * patch_size[1]-1)] = black
    plt.imshow(mask.permute(1, 2, 0))
    plt.show()


