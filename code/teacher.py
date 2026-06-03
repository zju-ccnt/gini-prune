import torch
from copy import deepcopy



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


from en_wiki import en_wiki_selected
calib_texts = en_wiki_selected  # 你给的那几句也行，建议多一些更稳（几十~几百句）

from fold_lowrank import fold_deleted_layers_into_next_layer_lowrank
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import AutoModelForCausalLM, AutoTokenizer
rank = 32
merge_layers = 10
batch_size = 4
#llama_13B_path = '~/.cache/modelscope/hub/models/AI-ModelScope/Mistral-7B-v0.1'
#llama_13B_path = '~/.cache/modelscope/hub/models/Qwen/Qwen2-7B'
llama_13B_path = '/home/ysf/.cache/modelscope/hub/models/LLM-Research/llama-2-7b'
save_path = '/home/ysf/Ours/teacher/model/qwen2-7b-6-0'
save_path = f'/home/ysf/Ours/teacher/model/llama2-7b-9-{rank}'
llama_13B_path = os.path.expanduser(llama_13B_path)
llama_13B = AutoModelForCausalLM.from_pretrained(llama_13B_path, trust_remote_code=True)
llama_13B_copy_to_compress = deepcopy(llama_13B)
tokenizer = AutoTokenizer.from_pretrained(llama_13B_path, trust_remote_code=True)
del_layer = [20]#[27, 22, 10] #20
for i in del_layer:
    llama_13B_copy_to_compress = fold_deleted_layers_into_next_layer_lowrank(
            teacher_model=llama_13B,
            student_model=llama_13B_copy_to_compress,
            tokenizer=tokenizer,
            base_layer_idx=i,
            merge_layers=merge_layers,          # 删 i+1..i+3，fold 到 i+4
            calib_texts=calib_texts, # 建议至少几十到几百句，否则UV会噪
            rank=rank,                  # 可试 8/16
            uv_steps=6000,            # 可试 200~800
            uv_batch_tokens=2048,
            uv_lr=5e-5, #5e-5
            uv_reg=5e-5, #5e-5
            device_for_collect="cpu",
            device_for_fit="cuda",   # 没GPU就改成 "cpu"
            max_length=256,
            batch_size=batch_size,   #1
            max_token_samples=800000,
            center=False,
    )


llama_13B_copy_to_compress.config.num_hidden_layers = len(llama_13B_copy_to_compress.model.layers)
llama_13B_copy_to_compress
llama_13B_copy_to_compress.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"模型已保存到: {save_path}")