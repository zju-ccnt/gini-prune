import torch
import torch.nn as nn
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# 加载模型
#llama_13B_path = '~/.cache/modelscope/hub/models/LLM-Research/llama-2-7b'
# ~/.cache/modelscope/hub/models/baichuan-inc/Baichuan2-7B-Base
# ~/.cache/modelscope/hub/models/Qwen/Qwen3-8B


# 设置设备
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 导入测试数据
from en_wiki import en_wiki_selected

class LayerSVD_Analyzer:
    """基于SVD的层重要性分析器"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.hooks = []
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def register_hooks(self):
        """注册hook以捕获每一层的输出"""
        self.hooks.clear()
        self.layer_outputs.clear()
        self.layer_inputs.clear()
        
        def get_hook(layer_name):
            def hook(module, input, output):
                # 对于LLaMA，output通常是tuple，第一个元素是hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                if isinstance(input, tuple):
                    x_in = input[0]
                else:
                    x_in = input
                
                if len(hidden_states.shape) == 3:
                    # [batch, seq_len, hidden_dim]
                    # 使用所有token的平均值，而不是只取第一个token
                    self.layer_outputs[layer_name] = hidden_states.mean(dim=1).detach().cpu()
                else:
                    self.layer_outputs[layer_name] = hidden_states.detach().cpu()
                if len(x_in.shape) == 3:
                    # [batch, seq_len, hidden_dim]
                    # 使用所有token的平均值，而不是只取第一个token
                    self.layer_inputs[layer_name] = x_in.mean(dim=1).detach().cpu()
                else:
                    self.layer_inputs[layer_name] = x_in.detach().cpu()
            return hook
        
        # 注册LLaMA每一层的钩子
        if hasattr(self.model.model, 'layers'):
            for i, layer in enumerate(self.model.model.layers):
                layer_name = f'layer_{i}'
                hook = layer.register_forward_hook(get_hook(layer_name))
                self.hooks.append(hook)
        elif hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
            for i, layer in enumerate(self.model.model.decoder.layers):
                layer_name = f'layer_{i}'
                hook = layer.register_forward_hook(get_hook(layer_name))
                self.hooks.append(hook)
        else:
            raise AttributeError(f"Cannot find layers in model structure: {dir(self.model.model)}")
            
        # 注册输入嵌入层
        # embed_hook = self.model.model.embed_tokens.register_forward_hook(get_hook('embedding'))
        # self.hooks.append(embed_hook)
        
        # 注册输出层（最后一层norm后的结果）
        def output_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.layer_outputs['output'] = hidden_states.mean(dim=1).detach().cpu()
            
        # output_hook = self.model.model.norm.register_forward_hook(output_hook)
        # self.hooks.append(output_hook)
        
    def remove_hooks(self):
        """移除钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def compute_activation_perplexity(self, activations):
        """
        activations: [N, D]
        """
        # 数值稳定
        x = activations.float()
        
        # 计算概率分布（按 feature）
        probs = torch.softmax(x, dim=-1)   # [N, D]
        
        # entropy per sample
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [N]
        
        # 平均
        entropy_mean = entropy.mean()
        
        perplexity = torch.exp(entropy_mean)
        
        return perplexity
        
    def collect_activations(self, texts, max_length=512, batch_size=4):
        """收集模型在测试文本上的激活值"""
        self.register_hooks()
        
        # 初始化存储结构
        all_layer_outputs = {}
        all_layer_inputs = {}
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
            batch_texts = texts[i:i+batch_size]
            
            # 编码文本并确保在正确的设备上
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=max_length
            )
            
            # 将输入移动到模型所在设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 清空当前batch的输出
            self.layer_outputs.clear()
            self.layer_inputs.clear()
            
            # 前向传播
            with torch.no_grad():
                try:
                    _ = self.model(**inputs)
                except RuntimeError as e:
                    print(f"Error in forward pass: {e}")
                    continue
            
            # 保存当前batch的各层输出
            for layer_name, output in self.layer_outputs.items():
                if layer_name not in all_layer_outputs:
                    all_layer_inputs[layer_name] = []
                    all_layer_outputs[layer_name] = []
                all_layer_inputs[layer_name].append(self.layer_inputs[layer_name])
                all_layer_outputs[layer_name].append(output)
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        self.remove_hooks()
        
        # 检查是否收集到了数据
        if not all_layer_outputs:
            raise RuntimeError("No activations were collected. Please check the model and input data.")
        
        # 合并所有batch的输出
        merged_outputs = {}
        merged_inputs = {}
        for layer_name, outputs in all_layer_outputs.items():
            if outputs:
                merged_outputs[layer_name] = torch.cat(outputs, dim=0)
                merged_inputs[layer_name] = torch.cat(all_layer_inputs[layer_name], dim=0)
                print(f"Layer {layer_name}: collected {merged_outputs[layer_name].shape[0]} samples")
            
        return merged_outputs, merged_inputs

    def compute_cosine_similarity(self, x_in, x_out):
        """
        x_in, x_out: [N, T, D] or [N, D]
        """
        # 展平 token 维度
        if x_in.dim() == 3:
            x_in = x_in.reshape(-1, x_in.size(-1))
            x_out = x_out.reshape(-1, x_out.size(-1))

        # 归一化
        x_in = torch.nn.functional.normalize(x_in, dim=-1)
        x_out = torch.nn.functional.normalize(x_out, dim=-1)

        cos = torch.sum(x_in * x_out, dim=-1)  # [N]

        return cos.mean()
    
    def compute_svd_metrics(self, activations, inputs=None):
        """计算SVD相关指标"""
        # 确保数据是浮点型
        activations = activations.float()
        
        # 中心化数据
        X = activations - activations.mean(dim=0, keepdim=True)
        
        # 使用更稳定的SVD计算方式
        try:
            metrics = {}
            cov = torch.matmul(X.T, X) / (X.shape[0] - 1)
            
            # SVD
            U, S, V = torch.svd(cov)
            
            # 计算基尼系数
            sorted_S = torch.sort(S)[0]  # 升序
            n = len(sorted_S)
            index = torch.arange(1, n + 1, device=S.device)

            gini = (torch.sum((2 * index - n - 1) * sorted_S)) / (n * torch.sum(sorted_S))
            metrics['gini_coefficient'] = gini.item()
            
            #perplexity = self.compute_activation_perplexity(X)
            #metrics['perplexity'] = perplexity.item()

            #cosine = self.compute_cosine_similarity(inputs, activations)
            #metrics['cosine'] = cosine.item()
            
            return metrics
            
        except Exception as e:
            print(f"Error in SVD computation: {e}")
            return None
    
    def analyze_all_layers(self, texts, max_length=512, batch_size=4):
        """分析所有层的SVD指标"""
        print(f"Analyzing {len(texts)} texts...")
        
        # 收集激活值
        activations, inputs = self.collect_activations(texts, max_length, batch_size)
        
        # 计算每层的指标
        layer_metrics = {}
        for layer_name, act in tqdm(activations.items(), desc="Computing SVD metrics"):
            metrics = self.compute_svd_metrics(act, inputs[layer_name])
            if metrics is not None:
                layer_metrics[layer_name] = metrics
                #print(f"{layer_name}: Gini = {metrics['gini_coefficient']} AND Perplexity = {metrics['perplexity']} AND Cosine = {metrics['cosine']}")
            
        return layer_metrics
    
    def choose_layers_to_be_pruner(self, layer_metrics, pruner = 8, protect_head = 5, protect_tail = 2):
        """选择要剪枝的连续层
        
        Args:
            layer_metrics: 层指标字典
            pruner: 要剪枝的层数量
            protect_head: 保护前protect_head层不被剪枝
            protect_tail: 保护后protect_tail层不被剪枝
        
        Returns:
            pruned_layers: 被剪枝的层下标数组
        """
        # 获取排序后的层（按重要性从高到低）
        scores = self.rank_layers_by_importance(layer_metrics)
        
        # 提取所有层名称和对应的分数
        all_layers = [name for name in layer_metrics.keys() if name.startswith('layer_')]
        all_layers.sort(key=lambda x: int(x.split('_')[1]))  # 按层索引排序
        
        # 获取层索引
        layer_indices = [int(layer.split('_')[1]) for layer in all_layers]
        total_layers = len(all_layers)
        # 确定可选的层范围（排除保护的头和尾）
        start_idx = protect_head
        end_idx = total_layers - protect_tail - pruner + 1
        
        if end_idx <= start_idx:
            print(f"警告：保护层过多，无法找到足够的连续层进行剪枝")
            return []
        mean_score = 0
        # 寻找平均值最小的连续pruner个层
        if True:
            print(f"寻找连续最小的连续值")
            min_avg_score = float('inf')
            best_start_position = -1
            
            for i in range(start_idx, end_idx):
                # 计算连续pruner个层的平均分数
                segment_scores = [scores[all_layers[j]] for j in range(i, i + pruner)]
                avg_score = sum(segment_scores) / pruner
                
                if avg_score < min_avg_score:
                    min_avg_score = avg_score
                    best_start_position = i
            mean_score = min_avg_score
        else:
            print(f"寻找连续最大的连续值")
            max_avg_score = float('-inf')
            best_start_position = -1
            
            for i in range(start_idx, end_idx):
                # 计算连续pruner个层的平均分数
                segment_scores = [scores[all_layers[j]] for j in range(i, i + pruner)]
                avg_score = sum(segment_scores) / pruner
                
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
                    best_start_position = i
            mean_score = max_avg_score
        
        # 获取被剪枝的层下标
        pruned_layer_names = all_layers[best_start_position:best_start_position + pruner]
        pruned_indices = [int(layer.split('_')[1]) for layer in pruned_layer_names]
        
        # 打印结果
        print(f"总层数: {total_layers}")
        print(f"保护前{protect_head}层: {layer_indices[:protect_head]}")
        print(f"保护后{protect_tail}层: {layer_indices[-protect_tail:]}")
        print(f"选中的连续{pruner}个层的起始位置: {best_start_position}")
        print(f"选中的连续{pruner}个层的实际索引: {pruned_indices}")
        print(f"平均重要性分数: {mean_score:.6f}")
        
        return pruned_indices
    
    def rank_layers_by_importance(self, layer_metrics):
        """根据重要性指标对层进行排序"""
        scores = {}
        
        # 只对实际的层（layer_X）进行排序，排除embedding和output
        layer_names = [name for name in layer_metrics.keys() if name.startswith('layer_')]
        
        for layer_name in layer_names:
            metrics = layer_metrics[layer_name]
            ## perplexity  gini_coefficient  cosine
            score = metrics['gini_coefficient']
                
            scores[layer_name] = score
        
        return scores
    
    def plot_layer_analysis(self, layer_metrics, save_path=None):
        # 提取层名称和指标
        layers = sorted([l for l in layer_metrics.keys() if l.startswith('layer_')], 
                        key=lambda x: int(x.split('_')[1]))
        
        if not layers:
            print("No layer data to plot")
            return
        
        # 准备数据
        gini_coeffs = [layer_metrics[l]['gini_coefficient'] for l in layers]
        
        layer_indices = [int(l.split('_')[1]) for l in layers]
        
        # 创建子图
        fig, axes = plt.subplots(1, 1, figsize=(15, 10))
        
        
        # 基尼系数
        ax2 = axes
        ax2.plot(layer_indices, gini_coeffs, 'r-o', markersize=4)
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Gini Coefficient')
        #ax2.set_title('Gini Coefficient (higher = more concentrated)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()
        
    def print_layer_ranking(self, sorted_layers, top_k=None):
        """打印层的重要性排序"""
        print("\n" + "="*70)
        print("Layer Importance Ranking (from most to least important)")
        print("="*70)
        
        for i, (layer_name, score) in enumerate(sorted_layers):
            if top_k and i >= top_k:
                break
            layer_idx = layer_name.split('_')[1]
            print(f"{i+1:3d}. Layer {layer_idx:3s} : importance score = {score:.6f}")
            
    def get_pruning_candidates(self, sorted_layers, prune_ratio=0.2):
        """获取可以剪枝的层候选"""
        num_layers = len(sorted_layers)
        num_prune = int(num_layers * prune_ratio)
        
        # 最不重要的层在列表末尾
        pruning_candidates = [layer for layer, _ in sorted_layers[-num_prune:]]
        
        print(f"\nPruning candidates ({prune_ratio*100:.0f}%):")
        for layer in pruning_candidates:
            layer_idx = layer.split('_')[1]
            print(f"  Layer {layer_idx}")
            
        return pruning_candidates

def plot_all_models_comparison(all_models_metrics, save_path=None):
    """
    将所有模型的 Gini 系数绘制在同一张折线图中
    
    Args:
        all_models_metrics: 字典，key为模型名称，value为该模型的layer_metrics
        save_path: 保存路径
    """
    plt.figure(figsize=(20, 10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    # 为每个模型绘制折线
    for idx, (model_name, layer_metrics) in enumerate(all_models_metrics.items()):
        # 提取层名称和指标
        layers = sorted([l for l in layer_metrics.keys() if l.startswith('layer_')], 
                        key=lambda x: int(x.split('_')[1]))
        
        if not layers:
            print(f"No layer data for {model_name}")
            continue
        
        # 准备数据 perplexity gini_coefficient cosine
        gini_coeffs = [layer_metrics[l]['gini_coefficient'] for l in layers]
        layer_indices = [int(l.split('_')[1]) for l in layers]
        marker = markers[idx % len(markers)]
        # 绘制折线图
        plt.plot(layer_indices, gini_coeffs, marker=marker, markersize=8, 
                linewidth=2, label=model_name)
    
    #plt.yscale('log')
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Cosine', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    #plt.title('Perplexity Comparison Across Different Models\n(higher = more concentrated)', 
    #          fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # 可选：添加一些辅助信息
    #plt.text(0.02, 0.98, 'Note: Lower Gini coefficient indicates more uniform distribution\nand potentially more important layers', 
    #         transform=plt.gca().transAxes, fontsize=10,
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comprehensive comparison plot saved to {save_path}")
    
    plt.show()


# 使用示例
from args import total_path, protect_tail, protect_head, pruner, total_rank, choose_model, only_pruner
def main():
    total_information = {} 

    model_name = choose_model
    model_path = total_path[model_name] 
    print("\n" + "="*70)
    print(f"Processing model: {model_name}")
    print(f"Model path: {model_path}")
    print("="*70)
    import os
    expanded_path = os.path.expanduser(model_path)
        
    model = AutoModelForCausalLM.from_pretrained(
         expanded_path, 
        trust_remote_code=True,
        load_in_8bit=False
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(expanded_path, trust_remote_code=True)
    analyzer = LayerSVD_Analyzer(model, tokenizer, device)
        
        # 使用少样本数据进行测试
    test_texts = en_wiki_selected[:50]  
    print(f"Using {len(test_texts)} test samples")
        
        # 分析所有层
    print("\nAnalyzing layers with SVD...")
    layer_metrics = analyzer.analyze_all_layers(
        test_texts,
        max_length=128, 
        batch_size=4
    )
        
        
    # 根据不同标准对层进行排序
    print("\n" + "="*70)
    print(f"Ranking layers for {model_name} by Gini Coefficient")
    print("="*70)
        
    criteria = [
        ('gini_coefficient', 'Gini Coefficient (lower = more important)')
    ]
        
    for criterion, description in criteria:
        print(f"\n--- {description} ---")
        total_information[model_name] = analyzer.choose_layers_to_be_pruner(
            layer_metrics,
            pruner = pruner[model_name],
            protect_head = protect_head,
            protect_tail = protect_tail[model_name]
        )
        
    print("\n" + "="*70)
    print("Start pruner")
    print("="*70)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    base = total_information[model_name][0] - 1
    number = len(total_information[model_name]) + 1
    print(f"base is {base} and pruner number is {number - 1}")
    from copy import deepcopy
    from fold_lowrank import fold_deleted_layers_into_next_layer_lowrank
    model_copy = deepcopy(model)
    calib_texts = en_wiki_selected 
    rank = total_rank[model_name]
    if only_pruner is True:
        rank = 0
    model_copy_to_compress = fold_deleted_layers_into_next_layer_lowrank(
            teacher_model= model,
            student_model= model_copy,
            tokenizer = tokenizer,
            base_layer_idx = base,
            merge_layers = number,          # 删 i+1..i+3，fold 到 i+4
            calib_texts= calib_texts, 
            rank = rank,         
            uv_steps= 6000,   
            uv_batch_tokens = 2048,
            uv_lr = 5e-5,
            uv_reg = 5e-5, 
            device_for_collect = "cpu",
            device_for_fit = "cuda",   # 没GPU就改成 "cpu"
            max_length=256,
            batch_size= 4,   
            max_token_samples=800000,
            center=False,
    )
    from pathlib import Path
    save_path = Path.cwd() / 'model' / f'{model_name}-cut{number-1}-fit{rank}'
    actual_layers = len(model_copy_to_compress.model.layers)
    model_copy_to_compress.config.num_hidden_layers = actual_layers
    if hasattr(model_copy_to_compress.config, 'num_layers'):
        original_layer_types = model_copy_to_compress.config.layer_types
        model_copy_to_compress.config.layer_types = original_layer_types[:actual_layers]
    save_path.mkdir(parents=True, exist_ok=True)
    model_copy_to_compress.save_pretrained(str(save_path))
    tokenizer.save_pretrained(save_path)
    print(f"模型已保存到: {save_path}")

if __name__ == "__main__":
    main()