import json
import pickle
import hashlib
import os
from typing import Dict, Tuple, Optional, Any
import torch
import time

class HiddenStatesCache:
    def __init__(self, cache_file: str = "cache.json"):
        self.cache_file = cache_file
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """加载缓存文件"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"警告: 缓存文件 {self.cache_file} 损坏，重新创建")
                return {}
        return {}
    
    def _save_cache(self):
        """保存缓存到文件"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"保存缓存失败: {e}")
    
    def _generate_cache_key(self, 
                           teacher_model: Any,
                           tokenizer: Any,
                           texts: list,
                           x_idx: int,
                           y_idx: int,
                           max_length: int,
                           max_token_samples: int) -> str:
        """生成唯一的缓存键"""
        # 生成文本内容的哈希
        text_hash = hashlib.md5(
            "|".join(texts[:min(10, len(texts))]).encode('utf-8')
        ).hexdigest()[:8]
        
        # 生成模型配置的标识
        model_config = {
            "model_name": teacher_model.__class__.__name__,
            "x_idx": x_idx,
            "y_idx": y_idx,
            "max_length": max_length,
            "max_token_samples": max_token_samples,
            "num_texts": len(texts)
        }
        
        model_hash = hashlib.md5(
            json.dumps(model_config, sort_keys=True).encode('utf-8')
        ).hexdigest()[:8]
        
        return f"{model_hash}_{text_hash}"
    
    def get_cached_hidden_states(self, cache_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取缓存的隐藏状态"""
        if cache_key not in self.cache_data:
            return None
        
        cache_info = self.cache_data[cache_key]
        x_path = cache_info.get("x_path")
        y_path = cache_info.get("y_path")
        
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            print(f"缓存文件不存在，清除缓存记录")
            self.clear_cache(cache_key)
            return None
        
        try:
            X_cpu = torch.load(x_path, map_location='cpu', weights_only=False)
            Y_cpu = torch.load(y_path, map_location='cpu', weights_only=False)
            print(f"[Cache] 从缓存加载隐藏状态: {cache_key}")
            return X_cpu, Y_cpu
        except Exception as e:
            print(f"加载缓存文件失败: {e}，重新收集")
            self.clear_cache(cache_key)
            return None
    
    def save_hidden_states(self, 
                          cache_key: str,
                          X_cpu: torch.Tensor,
                          Y_cpu: torch.Tensor,
                          save_dir: str = "./hidden_states_cache"):
        """保存隐藏状态到缓存"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存张量到文件
        x_path = os.path.join(save_dir, f"{cache_key}_X.pt")
        y_path = os.path.join(save_dir, f"{cache_key}_Y.pt")
        
        torch.save(X_cpu, x_path)
        torch.save(Y_cpu, y_path)
        
        # 更新缓存索引
        self.cache_data[cache_key] = {
            "x_path": x_path,
            "y_path": y_path,
            "x_shape": list(X_cpu.shape),
            "y_shape": list(Y_cpu.shape),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self._save_cache()
        print(f"[Cache] 隐藏状态已缓存: {cache_key}")
    
    def clear_cache(self, cache_key: str = None):
        """清除缓存"""
        if cache_key is None:
            # 清除所有缓存
            for key, info in list(self.cache_data.items()):
                for path_key in ["x_path", "y_path"]:
                    if path_key in info and os.path.exists(info[path_key]):
                        os.remove(info[path_key])
            self.cache_data = {}
        elif cache_key in self.cache_data:
            # 清除指定缓存
            info = self.cache_data[cache_key]
            for path_key in ["x_path", "y_path"]:
                if path_key in info and os.path.exists(info[path_key]):
                    os.remove(info[path_key])
            del self.cache_data[cache_key]
        
        self._save_cache()
