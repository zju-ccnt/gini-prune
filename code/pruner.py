import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from transformers import AutoModelForCausalLM, AutoTokenizer


def remove_merged_layers(model, layers_to_remove):
    """
    删除已经合并的层
    
    参数:
        model: 模型
        layers_to_remove: 要删除的层索引列表
    """
    # 按降序排序，从后往前删除
    layers_to_remove = sorted(layers_to_remove, reverse=True)
    
    for layer_idx in layers_to_remove:
        if layer_idx < len(model.model.layers):
            del model.model.layers[layer_idx]

from copy import deepcopy
def merge_layers_return_model(model, merge_base_lay, merge_layer_num):
   
    if hasattr(model.model, 'layers'):
        merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)
    elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        merge_layer_num = min(merge_layer_num, len(model.model.decoder.layers) - merge_base_lay - 1)
    else:
        raise AttributeError(f"Cannot find layers in model structure: {dir(model.model)}")
    
    model_copy = model
                       
    for diff_lay in range(merge_base_lay+merge_layer_num, merge_base_lay, -1):
        print(f"[Pruner] Pruner layer {diff_lay}")
        if hasattr(model.model, 'layers'):
            del(model_copy.model.layers[diff_lay])
        elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            del(model_copy.model.decoder.layers[diff_lay])
        else:
            raise AttributeError(f"Cannot find layers in model structure: {dir(model.model)}")
        
    return model_copy
