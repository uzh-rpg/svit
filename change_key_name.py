import re
from collections import OrderedDict

def change_key_name_from_timm_to_evo(input_dict):
    output_dict = OrderedDict()
    for old_key, value in input_dict.items():
        if re.search(r"^backbone\.blocks\.(\d+)\.norm1\.weight$", old_key):
            block_num = re.match(r"^backbone\.blocks\.(\d+)\.norm1\.weight$", old_key).group(1)
            new_key = "backbone.norms.{}.weight".format(block_num)
        elif re.search(r"^backbone\.blocks\.(\d+)\.norm1\.bias$", old_key):
            block_num = re.match(r"^backbone\.blocks\.(\d+)\.norm1\.bias$", old_key).group(1)
            new_key = "backbone.norms.{}.bias".format(block_num)
        elif re.search(r"^backbone\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)$", old_key):
            block_num = re.match(r"^backbone\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)$", old_key).group(1)
            weight_bias = re.match(r'^backbone\.blocks\.\d+\.attn\.qkv\.(weight|bias)$', old_key).group(1)
            qk_value = value[:value.shape[0]*2//3]
            v_value = value[value.shape[0]*2//3:]
            value = qk_value
            new_key = "backbone.qks.{}.qk.{}".format(block_num, weight_bias)
            addition_key = "backbone.vs.{}.v.{}".format(block_num, weight_bias)
            output_dict[addition_key] = v_value
        elif re.search(r"^backbone\.blocks\.(\d+)\.attn\.proj\.(weight|bias)$", old_key):
            block_num = re.match(r"^backbone\.blocks\.(\d+)\.attn\.proj\.(weight|bias)$", old_key).group(1)
            weight_bias = re.match(r'^backbone\.blocks\.\d+\.attn\.proj\.(weight|bias)$', old_key).group(1)
            new_key = "backbone.projs.{}.{}".format(block_num, weight_bias)
        else:
            new_key = old_key
        output_dict[new_key] = value

    return output_dict


revise_keys = [(r'blocks.\d+.', r'\g<0>TransformerEncoderLayer.'),
               ('attn.qkv.weight', 'self_attn.in_proj_weight'),
               ('attn.qkv.bias', 'self_attn.in_proj_bias'),
               ('attn.proj', 'self_attn.out_proj'),
               ('mlp.fc1', 'linear1'),
               ('mlp.fc2', 'linear2')]