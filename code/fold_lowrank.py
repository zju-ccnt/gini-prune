import torch

# -----------------------------
# 1) 把 (I + U V^T) 折叠到 next layer
# -----------------------------
@torch.no_grad()
def fold_lowrank_into_next_layer(next_layer, U, V):
    """
    将 T = I + U V^T 折叠到 next_layer 中吃 hidden state 的投影：
      - self_attn: q_proj, k_proj, v_proj
      - mlp: gate_proj, up_proj

    对每个线性层 y = W x：
      W <- W T = W + (W U) V^T
    """
    device = next_layer.self_attn.q_proj.weight.device if hasattr(next_layer, "self_attn") else U.device
    dtype = next_layer.self_attn.q_proj.weight.dtype if hasattr(next_layer, "self_attn") else U.dtype

    U = U.to(device=device, dtype=torch.float32)
    V = V.to(device=device, dtype=torch.float32)

    def fold_linear(linear):
        if linear is None:
            return
        W = linear.weight.data  # [out, in]
        # 计算: W += (W @ U) @ V^T
        # 用 float32 做乘法更稳，然后再 cast 回原 dtype
        W_f = W.float()
        delta = (W_f @ U) @ V.t()     # [out, in]
        W_f.add_(delta)
        W.copy_(W_f.to(dtype))

    attn = getattr(next_layer, "self_attn", None)
    if attn is not None:
        fold_linear(getattr(attn, "q_proj", None))
        fold_linear(getattr(attn, "k_proj", None))
        fold_linear(getattr(attn, "v_proj", None))

    #mlp = getattr(next_layer, "mlp", None)
    #if mlp is not None:
    #    fold_linear(getattr(mlp, "gate_proj", None))
    #    fold_linear(getattr(mlp, "up_proj", None))

    # 不折叠 o_proj / down_proj：
    # 它们的输入不是 "hidden state"，而分别是 attn 输出 / MLP 中间态。


# -----------------------------
# 2) 从 teacher 收集边界 hidden 对 (X,Y)
#    复用你之前 diag 版本的 _collect_hidden_pairs 即可
# -----------------------------
@torch.no_grad()
def _collect_hidden_pairs(
    teacher_model,
    tokenizer,
    texts,
    x_idx,
    y_idx,
    device="cuda",
    max_length=256,
    batch_size=2,
    max_token_samples=20000,
):
    teacher_model.eval().to(device)
    print(f"[Collect] Begin collect hidden pairs")
    X_list, Y_list = [], []
    total_kept = 0
    tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("teacher_model 没有返回 hidden_states，请确认 output_hidden_states=True")

        X = hs[x_idx]  # [B,T,H]
        Y = hs[y_idx]  # [B,T,H]

        m = attention_mask.bool()
        X = X[m]  # [N,H]
        Y = Y[m]  # [N,H]

        if max_token_samples is not None:
            remain = max_token_samples - total_kept
            if remain <= 0:
                break
            if X.size(0) > remain:
                idx = torch.randperm(X.size(0), device=X.device)[:remain]
                X = X[idx]
                Y = Y[idx]

        X_list.append(X.detach().float().cpu())
        Y_list.append(Y.detach().float().cpu())
        total_kept += X.size(0)

    X_all = torch.cat(X_list, dim=0)  # [N,H] CPU float32
    Y_all = torch.cat(Y_list, dim=0)
    return X_all, Y_all

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
    return model
# -----------------------------
# 3) 拟合 low-rank: 让 (Y - X) ≈ (XU)V^T
#    只训练 U,V（rank很小）
# -----------------------------
def _fit_lowrank_UV(
    X_cpu,
    Y_cpu,
    rank=8,
    device="cuda",
    steps=400,
    batch_tokens=2048,
    lr=5e-2,
    weight_decay=0.0,
    reg_uv=1e-4,
    center=True,
    seed=0,
):
    """
    X_cpu, Y_cpu: [N,H] on CPU float32
    返回 U,V: [H,r] float32 (在 CPU 上)
    """
    torch.manual_seed(seed)

    X = X_cpu
    Y = Y_cpu

    if center:
        # 去均值能显著减小“需要 bias”的成分（LLaMA线性层没bias）
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        X = X - X_mean
        Y = Y - Y_mean

    N, H = X.shape
    r = rank

    # 初始化 U,V
    U = torch.nn.Parameter(torch.randn(H, r, device=device, dtype=torch.float32) * 0.02)
    V = torch.nn.Parameter(torch.randn(H, r, device=device, dtype=torch.float32) * 0.02)


    opt = torch.optim.AdamW([U, V], lr=lr, weight_decay=weight_decay)

    X_dev = X.to(device=device)
    Y_dev = Y.to(device=device)

    # 目标：Delta ≈ (XU)V^T
    for t in range(steps):
        idx = torch.randint(0, N, (min(batch_tokens, N),), device=device)
        Xb = X_dev[idx]      # [B,H]
        Yb = Y_dev[idx]      # [B,H]
        Db = Yb - Xb         # [B,H]

        pred = (Xb @ U) @ V.t()  # [B,H]
        #pred =  (V.t() @ U) @ Xb
        #loss = torch.mean((pred - Db) ** 2)
        loss = torch.mean(1 - torch.nn.functional.cosine_similarity(pred, Db, dim=1))  # [B]
        # 轻微正则，防止 U,V 爆
        #loss = loss + reg_uv * (U.pow(2).mean() + V.pow(2).mean())
        #loss = torch.nn.functional.cosine_embedding_loss(pred, Db, torch.ones(pred.size(0), device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # 可选：打印
        if (t) % 1000 == 0:
            print(f"  [UV fit] step {t}/{steps}, loss={loss.item():.6f}")

    U_cpu = U.detach().float().cpu()
    V_cpu = V.detach().float().cpu()
    return U_cpu, V_cpu

from cache import HiddenStatesCache

# -----------------------------
# 4) 主函数：估计 UV -> 折叠到 next -> 删除层
# -----------------------------
def fold_deleted_layers_into_next_layer_lowrank(
    teacher_model,
    student_model,
    tokenizer,
    base_layer_idx: int,
    merge_layers: int = 4,
    calib_texts=None,
    rank: int = 8,
    uv_steps: int = 400,
    uv_batch_tokens: int = 2048,
    uv_lr: float = 5e-2,
    uv_reg: float = 1e-4,
    device_for_collect: str = "cpu",   # 你现在在CPU上跑teacher，可以保持一致
    device_for_fit: str = "cuda",      # 建议用GPU拟合UV；没有GPU可设为"cpu"
    max_length: int = 256,
    batch_size: int = 2,
    max_token_samples: int = 20000,
    center: bool = True,
):
    """
    删除 base+1..base+merge_layers-1，并把等效 low-rank 变换折叠到 base+merge_layers (next layer)。
    """
    print(f"[Fold] Entry")
    
    if calib_texts is None or len(calib_texts) == 0:
        raise ValueError("calib_texts 不能为空（用来估计等效变换）")

    if student_model is None:
        raise ValueError("student_model 为 None（你可能遇到了 remove_merged_layers 返回 None 的情况）")

    layers = student_model.model.layers
    total_layers = len(layers)

    next_idx = base_layer_idx + merge_layers
    del_start = base_layer_idx + 1
    del_end = base_layer_idx + merge_layers - 1
        

    if merge_layers <= 1:
        print("merge_layers <= 1, 无需 folding")
        return student_model
    if next_idx >= total_layers:
        print(f"next layer idx={next_idx} 超出范围(total={total_layers})，跳过")
        return student_model
    if del_start > del_end:
        print("没有要删除的层，跳过")
        return student_model

    # 在 teacher hidden_states 里：
    # hidden_states[0]=embed输出；hidden_states[i+1]=layer i 输出
    
    if rank != 0:
        x_hs_idx = base_layer_idx + 1
        y_hs_idx = base_layer_idx + merge_layers

        cache = HiddenStatesCache()
        cache_key = cache._generate_cache_key(
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            texts=calib_texts,
            x_idx=x_hs_idx,
            y_idx=y_hs_idx,
            max_length=max_length,
            max_token_samples=max_token_samples,
        )
        cached = cache.get_cached_hidden_states(cache_key)
        if cached is not None:
            print(f"[Cache] Use cache data")
            X_cpu, Y_cpu = cached
        # 1) 收集边界 (X,Y)
        else:
            print(f"[Fold] Begin collect hidden pairs")
            X_cpu, Y_cpu = _collect_hidden_pairs(
                teacher_model=teacher_model,
                tokenizer=tokenizer,
                texts=calib_texts,
                x_idx=x_hs_idx,
                y_idx=y_hs_idx,
                device=device_for_collect,
                max_length=max_length,
                batch_size=batch_size,
                max_token_samples=max_token_samples,
            )
            cache.save_hidden_states(cache_key=cache_key, X_cpu=X_cpu, Y_cpu=Y_cpu)
        print(f"[Fold] Begin fit lowrank UV")
        # 2) 拟合 U,V（只训练UV，很小）
        U_cpu, V_cpu = _fit_lowrank_UV(
            X_cpu=X_cpu,
            Y_cpu=Y_cpu,
            rank=rank,
            device=device_for_fit,
            steps=uv_steps,
            batch_tokens=uv_batch_tokens,
            lr=uv_lr,
            reg_uv=uv_reg,
            center=center,
        )

        # 3) 折叠到 next layer
        next_layer = layers[next_idx]
        fold_lowrank_into_next_layer(next_layer, U_cpu, V_cpu)
    else:
        pass

    # 4) 删除层（兼容 remove_merged_layers 返回 None / 返回 model）
    layers_to_delete = list(range(del_start, del_end + 1))
    print(f"[FOLD-LOWRANK+DEL] base={base_layer_idx} delete={layers_to_delete} fold_into_next={next_idx} rank={rank}")

    ret = remove_merged_layers(student_model, layers_to_delete)
    if ret is not None:
        student_model = ret

    return student_model
