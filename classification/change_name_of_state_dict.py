import re

import timm


def change_key_name_fn(state_dict, start_layer: int):
    return {k.replace('TransformerEncoderLayer', 'SelectiveTransformerEncoderLayer') if k[7:9].strip(
        '.').isnumeric() and int(k[7:9].strip('.')) >= start_layer else k: v for k, v in state_dict.items()}


def add_TransformerEncoderLayer_after_numbers(input_string):
  # Compile a regular expression pattern to match numbers in the input string
  pattern = re.compile(r'blocks.\d+.')

  # Find all numbers in the input string and add 'TransformerEncoderLayer' after each number
  output_string = pattern.sub(r'\g<0>TransformerEncoderLayer.', input_string)

  return output_string


def change_key_name_from_timm_to_pytorch(state_dict):
    new_dict= {}
    for key, value in state_dict.items():
        new_key = add_TransformerEncoderLayer_after_numbers(key).replace('attn.qkv.weight', 'self_attn.in_proj_weight')\
            .replace('attn.qkv.bias', 'self_attn.in_proj_bias').replace('attn.proj', 'self_attn.out_proj')\
            .replace('mlp.fc1', 'linear1').replace('mlp.fc2', 'linear2')
        new_dict[new_key] = value
    return new_dict

def change_key_name_from_timm_to_evo(input_dict):
    output_dict = {}
    for old_key, value in input_dict.items():
        if re.search(r"^blocks\.(\d+)\.norm1\.weight$", old_key):
            block_num = re.match(r"^blocks\.(\d+)\.norm1\.weight$", old_key).group(1)
            new_key = "norms.{}.weight".format(block_num)
        elif re.search(r"^blocks\.(\d+)\.norm1\.bias$", old_key):
            block_num = re.match(r"^blocks\.(\d+)\.norm1\.bias$", old_key).group(1)
            new_key = "norms.{}.bias".format(block_num)
        elif re.search(r"^blocks\.(\d+)\.attn\.qkv\.(weight|bias)$", old_key):
            block_num = re.match(r"^blocks\.(\d+)\.attn\.qkv\.(weight|bias)$", old_key).group(1)
            weight_bias = re.match(r'^blocks\.\d+\.attn\.qkv\.(weight|bias)$', old_key).group(1)
            qk_value = value[:value.shape[0]*2//3]
            v_value = value[value.shape[0]*2//3:]
            value = qk_value
            new_key = "qks.{}.qk.{}".format(block_num, weight_bias)
            addition_key = "vs.{}.v.{}".format(block_num, weight_bias)
            output_dict[addition_key] = v_value
        elif re.search(r"^blocks\.(\d+)\.attn\.proj\.(weight|bias)$", old_key):
            block_num = re.match(r"^blocks\.(\d+)\.attn\.proj\.(weight|bias)$", old_key).group(1)
            weight_bias = re.match(r'^blocks\.\d+\.attn\.proj\.(weight|bias)$', old_key).group(1)
            new_key = "projs.{}.{}".format(block_num, weight_bias)
        elif old_key in ['head.weight', 'head.bias', 'norm.weight', 'norm.bias']:
            new_key = old_key.replace('head', 'head_cls').replace('norm', 'norm_cls')
        else:
            new_key = old_key
        output_dict[new_key] = value

    return output_dict

if __name__ == '__main__':
    vit = timm.create_model('vit_tiny_patch16_224')
    nnvit = timm.create_model('nnvit_tiny_patch16_224')
    assert change_key_name_from_timm_to_pytorch(vit.state_dict()).keys() == nnvit.state_dict().keys()