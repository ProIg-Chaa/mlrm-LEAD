import os
import re
import torch
import torch.nn.functional as F
import random
import numpy as np
import math

a  = 1.0   # 锚点加权缩放：1.0=原样；0.0=禁用锚点加权
b1 = 1.0   # 触发 normal→soft 所需的“额外熵阈值” (越大越难切到 soft)
b2 = 0.2   # 触发 soft→normal 所需的“额外熵阈值” (越大越难切回 normal)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import transformers
        transformers.set_seed(seed)
    except Exception:
        pass


def apply_sampling_filter(logits, top_k=0, top_p=1.0, min_p=0.0):
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, float('-inf'), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        logits = torch.where(probs < min_p, float('-inf'), logits)
    return logits


def get_math_symbols_ids(tokenizer):
    math_symbols = [
        "+", "-", "*", "/", "^", "=", "<", ">", "\\leq", "\\geq", "\\neq", "\\approx", "\\sim", "\\equiv", "\\to", "\\implies", "\\iff",
        "(", ")", "[", "]", "{", "}", "\\left(", "\\right)", "\\left[", "\\right]", "\\left\\{", "\\right\\}",
        "\\begin{pmatrix}", "\\end{pmatrix}",
        "\\frac", "\\dfrac", "\\sqrt", "\\sqrt[]",
        "\\in", "\\notin", "\\subset", "\\supset", "\\subseteq", "\\supseteq", "\\cup", "\\cap", "\\emptyset", "\\varnothing",
        "\\pi", "\\theta", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\lambda", "\\mu", "\\nu",
        "\\sin", "\\cos", "\\tan", "\\arcsin", "\\arccos", "\\arctan", "\\log", "\\ln", "\\exp",
        "_", "\\binom", "\\choose", "\\cdot", "\\dots", "\\ldots", "\\cdots", "\\vdots", "\\ddots",
        "\\mathbb", "\\mathbf", "\\mathrm", "\\text", "\\mbox",
        "\\infty", "\\circ", "\\prime", "\\ast", "\\star", "\\triangle", "\\triangleleft", "\\triangleright", "\\perp", "\\parallel", "\\angle",
        "\\boxed", "\\overline", "\\underline", "\\lceil", "\\rceil", "\\lfloor", "\\rfloor", "\\left", "\\right", "\\mid", "|", "\\vert", "\\Vert",
        "\\because", "\\therefore", "\\forall", "\\exists", "\\wedge", "\\vee", "\\neg",
        "\\sum", "\\prod", "\\int", "\\lim", "\\min", "\\max", "\\arg", "\\deg", "\\gcd", "\\operatorname",
        "\\cot", 
        "\\cotg", "\\sec", "\\csc",
    ]
    math_symbols += [chr(c) for c in range(ord('0'), ord('9')+1)]
    math_symbols += [chr(c) for c in range(ord('a'), ord('z')+1)]
    math_symbols += [chr(c) for c in range(ord('A'), ord('Z')+1)]
    math_token_ids = set()
    for symbol in math_symbols:
        math_token_ids.update(tokenizer.encode(symbol, add_special_tokens=False))
    return math_token_ids
    

def generate_cot(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    vision_inputs = {}
    for key in list(kwargs.keys()):
        if any(tag in key for tag in ("pixel", "image", "video")):
            value = kwargs.pop(key)
            if value is not None:
                vision_inputs[key] = value

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)

    stream_callback = kwargs.pop("stream_callback", None)
    token_trace = kwargs.pop("token_trace", None)

    # ============================================

    batch_size = input_ids.shape[0]
    device = input_ids.device

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))

    generated = input_ids.clone()
    attn_mask = attention_mask.clone() if attention_mask is not None else None
    past_key_values = None
    cache_position = torch.arange(generated.shape[1], device=device, dtype=torch.long)
        
    for step in range(max_new_tokens):
        cur_batch = generated.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {"input_ids": generated}
            if attn_mask is not None:
                model_inputs["attention_mask"] = attn_mask
            if vision_inputs:
                model_inputs.update(vision_inputs)
            model_inputs["cache_position"] = cache_position
        else:
            if attn_mask is not None:
                attention_mask_new = torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device)
                attn_mask = torch.cat([attn_mask, attention_mask_new], dim=1)
            model_inputs = {
                "input_ids": next_tokens.unsqueeze(1),
                "past_key_values": past_key_values,
            }
            if attn_mask is not None:
                model_inputs["attention_mask"] = attn_mask
            model_inputs["cache_position"] = cache_position

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        if vision_inputs:
            vision_inputs = {}
        cache_position = cache_position[-1:] + 1

        next_token_logits = outputs.logits[:, -1, :]  # [cur_batch, vocab]
        raw_probs = F.softmax(next_token_logits, dim=-1)
        raw_entropy = -(
            raw_probs * raw_probs.clamp(min=1e-8).log()
        ).sum(dim=-1)
        logits = next_token_logits / temperature
        logits = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)

        probs = F.softmax(logits, dim=-1)
        filtered_entropy = -(
            probs * probs.clamp(min=1e-8).log()
        ).sum(dim=-1)
        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        for bi, orig in enumerate(unfinished_idx):
            token_id = next_tokens[bi].item()
            all_generated[orig].append(token_id)
            if token_trace is not None:
                token_trace.append({
                    "step": int(step),
                    "batch_index": int(orig),
                    "token_id": int(token_id),
                    "raw_entropy": float(raw_entropy[bi].item()),
                    "filtered_entropy": float(filtered_entropy[bi].item()),
                    "selected_prob": float(probs[bi, next_tokens[bi]].item()),
                    "mode": "normal",
                })
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])

        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

        if len(unfinished_idx) == 0:
            break
        generated = generated[keep_idx]
        next_tokens = next_tokens[keep_idx]
        if attention_mask is not None:
            attention_mask = attention_mask[keep_idx]
        if attn_mask is not None:
            attn_mask = attn_mask[keep_idx]
        keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=generated.device)
        if hasattr(past_key_values, "batch_select_indices"):
            past_key_values.batch_select_indices(keep_idx_tensor)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out


def generate_lead(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    vision_inputs = {}
    for key in list(kwargs.keys()):
        if any(tag in key for tag in ("pixel", "image", "video")):
            value = kwargs.pop(key)
            if value is not None:
                vision_inputs[key] = value

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)

    # ---- lead ----
    alpha_0                = kwargs.pop("alpha_0", 1.0) # adjustable
    beta_0                 = kwargs.pop("beta_0", 0.7)
    window_size            = kwargs.pop("window_size", 256) #“冷却时间 / 稳定窗口”，用来防止 soft 模式与 normal 模式之间的频繁来回震荡
    thinking_token_id      = kwargs.pop("thinking_token_id", None)
    end_thinking_token_id  = kwargs.pop("end_thinking_token_id", None)
    max_switch_count       = kwargs.pop("max_switch_count", None) # adjustable for efficiency
    math_ids_tensor        = kwargs.pop("math_ids_tensor", None)
    convergence_words      = kwargs.get("convergence_words", "</think>")
    termination_words      = kwargs.get("termination_words", "</think>\n\nThe final answer is")
    termination_max_tokens = kwargs.pop("termination_max_tokens", 32)

    stream_callback       = kwargs.pop("stream_callback", None)
    token_trace           = kwargs.pop("token_trace", None)

    # ============================================

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    batch_size, device = input_ids.shape[0], input_ids.device
    E = model.get_input_embeddings().weight  # [vocab_size, dim]
    def _resolve_token_id(token_text, fallback_text=None):
        token_id = None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_text)
        except Exception:
            token_id = None
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if token_id is None or token_id == tokenizer.unk_token_id or (isinstance(token_id, int) and token_id < 0):
            text = fallback_text if fallback_text is not None else token_text
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                token_id = encoded[0]
        if token_id is None:
            token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        return token_id
    if thinking_token_id is None:
        thinking_token_id = _resolve_token_id("<think>")
    if end_thinking_token_id is None:
        end_thinking_token_id = _resolve_token_id("</think>")

    #===把<think>替换成<|image_pad|>=======
    imgpad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    thinking_token_id = imgpad_id
    #=====================================

    start_thinking_emb, end_thinking_emb = E[thinking_token_id], E[end_thinking_token_id]
    newline_id = _resolve_token_id("\\n", "\n")
    line_break_emb = E[newline_id]
    past_key_values = None
    cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
        
    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size)) # bs >= 1 is supported
    mode = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0: soft, 1: normal
    mode_stay_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    locked_normal_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    if max_switch_count is not None:
        switch_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        convergence_ids = tokenizer.encode(convergence_words, add_special_tokens=False)
        termination_ids = tokenizer.encode(termination_words, add_special_tokens=False)
        injecting = torch.zeros(batch_size, dtype=torch.bool, device=device)
        inject_queues = [[] for _ in range(batch_size)]
        answer_budget = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        cur_batch = attention_mask.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {
                "input_ids": input_ids.clone(), 
            }
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if vision_inputs:
                model_inputs.update(vision_inputs)
            model_inputs["cache_position"] = cache_position
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {
                "inputs_embeds": last_emb.unsqueeze(1), 
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            model_inputs["cache_position"] = cache_position

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        if vision_inputs:
            vision_inputs = {}
        cache_position = cache_position[-1:] + 1
        
        logits_original = outputs.logits[:, -1, :]
        probs_original = F.softmax(logits_original, dim=-1)
        logits = logits_original / temperature  
        logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)  # [B, N, V]
        probs = F.softmax(logits_filtered, dim=-1)
        filtered_entropy = -(
            probs * probs.clamp(min=1e-8).log()
        ).sum(dim=-1)

        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)  # [B, N]
        locked_normal_mask = locked_normal_mask | (next_tokens == end_thinking_token_id)

        if max_switch_count is not None and injecting.any():
            mask_list = [injecting[i].item() and len(inject_queues[i]) > 0 for i in range(cur_batch)]
            force_mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
            if force_mask.any():
                force_toks = torch.tensor([inject_queues[i].pop(0) for i in range(cur_batch) if mask_list[i]], \
                                          device=device, dtype=torch.long)
                next_tokens[force_mask] = force_toks
            if injecting.any():
                done_mask = torch.tensor([injecting[i] and (len(inject_queues[i]) == 0) for i in range(cur_batch)], \
                                         device=device, dtype=torch.bool)
                injecting[done_mask] = False
        
        cur_entropy = -(probs_original * (probs_original.clamp(min=1e-8).log())).sum(dim=-1)
        if step == 0:
            cur_ref_entropy = cur_entropy.clone()
        else:
            mode_stay_steps += 1
            allow_switch = (mode_stay_steps >= window_size)
            to_normal = (mode == 0) & (cur_entropy < cur_ref_entropy) & (cur_entropy < b2)
            to_soft = (mode == 1) & (cur_entropy > cur_ref_entropy) & allow_switch & (~locked_normal_mask) & (cur_entropy > b1)

            
            if (to_normal.any() or to_soft.any()) and step % 50 == 0:  # 每50步打印一次
                print(f"[LEAD] Step {step}: to_normal={to_normal.nonzero().squeeze(-1).tolist()}, to_soft={to_soft.nonzero().squeeze(-1).tolist()}")
            
            mode[to_normal] = 1
            mode[to_soft] = 0
            mode_stay_steps[to_normal | to_soft] = 0
            cur_ref_entropy[to_normal | to_soft] = cur_entropy[to_normal | to_soft]

            #输出表明print
            if to_normal.any() or to_soft.any():
                print(f"[LEAD] Step {step}: switch → "
                    f"{'to_normal' if to_normal.any() else 'to_soft'} | "
                    f"entropy={cur_entropy.mean().item():.4f}")


            if max_switch_count is not None:
                switch_count = switch_count + to_normal.long() 
            
        is_normal = (mode == 1) | locked_normal_mask
        if math_ids_tensor is not None:
            is_math_token = (next_tokens.unsqueeze(-1) == math_ids_tensor).any(dim=-1)
            is_normal[is_math_token] = True
        is_soft = ~is_normal
        
        normal_emb = E[next_tokens]
        soft_emb = torch.matmul(probs_original, E)

        alpha = alpha_0 + (1 - alpha_0) * float(step) / float(max_new_tokens)
        if step == 0:
            soft_emb = 0.9 * soft_emb + 0.1 * line_break_emb
        else:
            mixed_emb = alpha * soft_emb + a * (1 - alpha) * start_thinking_emb
            
            soft_emb = torch.where(to_soft[:, None], mixed_emb, soft_emb)
        beta = beta_0 + (1 - beta_0) * float(step) / float(max_new_tokens)

        if step % 200 == 0:
            print(f"[LEAD] Step {step}: alpha={alpha:.3f}, beta={beta:.3f}, soft_ratio={(mode==0).float().mean():.2f}")


        if step > 0:
            mixed_emb = beta * soft_emb + (1 - beta) * end_thinking_emb
            normal_emb = torch.where(to_normal[:, None], mixed_emb, normal_emb)
        last_emb = torch.where(is_soft[:, None], soft_emb, normal_emb)

        if max_switch_count is not None and step > 0:
            trigger = (switch_count >= max_switch_count) & (switch_count <= 2 * max_switch_count) & to_normal
            
            if trigger.any():
                print(f"[LEAD] Inject convergence at step {step}, sample={trigger.nonzero().squeeze(-1).tolist()}")

            
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(convergence_ids)
                injecting = injecting | trigger

            trigger = (switch_count > 2 * max_switch_count) & to_normal

            if trigger.any():
                print(f"[LEAD] Inject termination at step {step}, sample={trigger.nonzero().squeeze(-1).tolist()}")


            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(termination_ids) 
                injecting = injecting | trigger 
                answer_budget[trigger] = termination_max_tokens
            active = (answer_budget >= 0)
            if active.any():
                answer_budget = torch.where(active, answer_budget - 1, answer_budget)

        for bi, orig in enumerate(unfinished_idx):
            token_id = next_tokens[bi].item()
            all_generated[orig].append(token_id)
            if token_trace is not None:
                token_trace.append({
                    "step": int(step),
                    "batch_index": int(orig),
                    "token_id": int(token_id),
                    "raw_entropy": float(cur_entropy[bi].item()),
                    "filtered_entropy": float(filtered_entropy[bi].item()),
                    "selected_prob": float(probs[bi, next_tokens[bi]].item()),
                    "mode": "soft" if bool(is_soft[bi].item()) else "normal",
                    "alpha": float(alpha),
                    "beta": float(beta),
                })
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])
        
        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        if max_switch_count is not None:
            budget_done = (answer_budget == 0) 
            cur_finished = cur_finished | budget_done

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]
        if len(unfinished_idx) == 0:
            break
        last_emb = last_emb[keep_idx]
        attention_mask = attention_mask[keep_idx]
        mode = mode[keep_idx]
        mode_stay_steps = mode_stay_steps[keep_idx]
        cur_ref_entropy = cur_ref_entropy[keep_idx]
        locked_normal_mask = locked_normal_mask[keep_idx]
        if hasattr(past_key_values, "batch_select_indices"):
            keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
            past_key_values.batch_select_indices(keep_idx_tensor)
        if max_switch_count is not None:
            switch_count = switch_count[keep_idx]
            injecting = injecting[keep_idx]
            inject_queues = [inject_queues[i] for i in keep_idx.tolist()]
            answer_budget = answer_budget[keep_idx]

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    return out

# 找出输入中的视觉token位置并掩盖其他token，得到视觉token掩码，供后续动态视觉锚点计算使用
def _build_visual_token_mask(input_ids, tokenizer):
    imgpad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if imgpad_id is None or imgpad_id < 0:
        return visual_mask

    batch_size, seq_len = input_ids.shape
    for bi in range(batch_size):
        inside_vision = False
        for pos in range(seq_len):
            token_id = int(input_ids[bi, pos].item())
            if token_id == vision_start_id:
                inside_vision = True
            elif token_id == vision_end_id:
                inside_vision = False
            elif inside_vision and token_id == imgpad_id:
                visual_mask[bi, pos] = True

    # Fallback: if the explicit vision span is unavailable, use raw image pad positions.
    if not visual_mask.any():
        visual_mask = input_ids == imgpad_id
    return visual_mask


def _compute_dynamic_visual_anchor(
    attn_layers,
    soft_emb,
    prompt_hidden_states,
    visual_token_mask,
    prompt_len,
    top_m,
):
    """
    Select top-m visual tokens from current-token attention, then pool them with
    latent embedding attention to build a dynamic visual anchor.
    """
    attn_device = next(layer.device for layer in attn_layers if layer is not None)
    soft_emb = soft_emb.to(attn_device)
    prompt_hidden_states = prompt_hidden_states.to(attn_device)
    visual_token_mask = visual_token_mask.to(attn_device)
    batch_size, hidden_size = soft_emb.shape
    anchors = soft_emb.clone()
    has_anchor = torch.zeros(batch_size, dtype=torch.bool, device=soft_emb.device)

    if not attn_layers:
        raise RuntimeError(
            "generate_lead_attenachor requires decoder attentions, "
            "but model outputs.attentions is empty. Load the model with "
            "attn_implementation='eager'."
        )

    layer_views = []
    for layer_attn in attn_layers:
        if layer_attn is None:
            continue
        if layer_attn.dim() != 4:
            raise RuntimeError(
                f"Unexpected attention tensor rank {layer_attn.dim()} "
                "in generate_lead_attenachor; expected [B, heads, q_len, kv_len]."
            )
        layer_views.append(layer_attn[:, :, -1, :].mean(dim=1))
    if not layer_views:
        raise RuntimeError("No usable attention layers found in outputs.attentions.")

    # Aggregate only the final few decoder layers, then mean over layers.
    current_attn = torch.stack(layer_views, dim=0).mean(dim=0)
    kv_len = current_attn.shape[-1]
    prompt_key_len = min(prompt_len, kv_len)

    for bi in range(batch_size):
        visual_positions = visual_token_mask[bi, :prompt_key_len].nonzero(as_tuple=False).squeeze(-1)
        if visual_positions.numel() == 0:
            continue

        visual_scores = current_attn[bi, visual_positions]
        cur_top_m = min(int(top_m), int(visual_positions.numel()))
        if cur_top_m <= 0:
            continue
        top_indices = torch.topk(visual_scores, k=cur_top_m, dim=0).indices
        top_visual_positions = visual_positions[top_indices]

        selected_visual_states = prompt_hidden_states[bi, top_visual_positions, :]
        latent_scores = torch.matmul(
            selected_visual_states,
            soft_emb[bi],
        ) / math.sqrt(hidden_size)
        latent_weights = F.softmax(latent_scores, dim=0)
        anchor = torch.sum(
            selected_visual_states * latent_weights.unsqueeze(-1),
            dim=0,
        )
        anchors[bi] = anchor
        has_anchor[bi] = True

    return anchors, has_anchor


def _get_text_attn_implementation(model):
    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
    ]
    values = []
    for config in configs:
        if config is not None and hasattr(config, "_attn_implementation"):
            values.append((config, config._attn_implementation))
    return values


def _set_text_attn_implementation(config_values, implementation):
    for config, _ in config_values:
        config._attn_implementation = implementation


def _restore_text_attn_implementation(config_values):
    for config, value in config_values:
        config._attn_implementation = value


def generate_lead_attenachor(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    vision_inputs = {}
    for key in list(kwargs.keys()):
        if any(tag in key for tag in ("pixel", "image", "video")):
            value = kwargs.pop(key)
            if value is not None:
                vision_inputs[key] = value

    # ---- **gen_kwargs ----
    temperature = kwargs.get("temperature", 1.0)
    top_p = kwargs.get("top_p", 1.0)
    top_k = kwargs.get("top_k", 0)
    min_p = kwargs.get("min_p", 0)
    max_new_tokens = kwargs.get("max_new_tokens", 32768)
    do_sample = kwargs.get("do_sample", True)

    # ---- lead ----
    alpha_0 = kwargs.pop("alpha_0", 1.0)
    beta_0 = kwargs.pop("beta_0", 0.7)
    window_size = kwargs.pop("window_size", 256)
    thinking_token_id = kwargs.pop("thinking_token_id", None)
    end_thinking_token_id = kwargs.pop("end_thinking_token_id", None)
    max_switch_count = kwargs.pop("max_switch_count", None)
    math_ids_tensor = kwargs.pop("math_ids_tensor", None)
    convergence_words = kwargs.get("convergence_words", "</think>")
    termination_words = kwargs.get("termination_words", "</think>\n\nThe final answer is")
    termination_max_tokens = kwargs.pop("termination_max_tokens", 32)
    visual_anchor_top_m = kwargs.pop("visual_anchor_top_m", 32)

    stream_callback = kwargs.pop("stream_callback", None)
    token_trace = kwargs.pop("token_trace", None)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    batch_size, device = input_ids.shape[0], input_ids.device
    E = model.get_input_embeddings().weight

    def _resolve_token_id(token_text, fallback_text=None):
        token_id = None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_text)
        except Exception:
            token_id = None
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if token_id is None or token_id == tokenizer.unk_token_id or (isinstance(token_id, int) and token_id < 0):
            text = fallback_text if fallback_text is not None else token_text
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                token_id = encoded[0]
        if token_id is None:
            token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        return token_id

    visual_token_mask = _build_visual_token_mask(input_ids, tokenizer)
    prompt_len = input_ids.shape[1]
    prompt_hidden_states = None

    if thinking_token_id is None:
        thinking_token_id = _resolve_token_id("<think>")
    if end_thinking_token_id is None:
        end_thinking_token_id = _resolve_token_id("</think>")

    imgpad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    thinking_token_id = imgpad_id

    end_thinking_emb = E[end_thinking_token_id]
    newline_id = _resolve_token_id("\\n", "\n")
    line_break_emb = E[newline_id]
    past_key_values = None
    cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))
    mode = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0: soft, 1: normal
    mode_stay_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    locked_normal_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if max_switch_count is not None:
        switch_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        convergence_ids = tokenizer.encode(convergence_words, add_special_tokens=False)
        termination_ids = tokenizer.encode(termination_words, add_special_tokens=False)
        injecting = torch.zeros(batch_size, dtype=torch.bool, device=device)
        inject_queues = [[] for _ in range(batch_size)]
        answer_budget = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        cur_batch = attention_mask.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {
                "input_ids": input_ids.clone(),
            }
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if vision_inputs:
                model_inputs.update(vision_inputs)
            model_inputs["cache_position"] = cache_position
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {
                "inputs_embeds": last_emb.unsqueeze(1),
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            model_inputs["cache_position"] = cache_position

        potential_to_soft = (
            past_key_values is not None
            and ((mode == 1) & ((mode_stay_steps + 1) >= window_size) & (~locked_normal_mask)).any()
        )
        need_decode_attn = bool(potential_to_soft)
        attn_config_values = None
        if need_decode_attn:
            attn_config_values = _get_text_attn_implementation(model)
            _set_text_attn_implementation(attn_config_values, "eager")
        try:
            with torch.no_grad():
                outputs = model(
                    **model_inputs,
                    use_cache=True,
                    output_attentions=need_decode_attn,
                    output_hidden_states=(prompt_hidden_states is None),
                )
        finally:
            if attn_config_values is not None:
                _restore_text_attn_implementation(attn_config_values)
        past_key_values = outputs.past_key_values
        if prompt_hidden_states is None:
            prompt_hidden_states = outputs.hidden_states[-1][:, :prompt_len, :].detach()
        if vision_inputs:
            vision_inputs = {}
        cache_position = cache_position[-1:] + 1

        logits_original = outputs.logits[:, -1, :]
        probs_original = F.softmax(logits_original, dim=-1)
        logits = logits_original / temperature
        logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)
        probs = F.softmax(logits_filtered, dim=-1)
        filtered_entropy = -(
            probs * probs.clamp(min=1e-8).log()
        ).sum(dim=-1)

        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)
        locked_normal_mask = locked_normal_mask | (next_tokens == end_thinking_token_id)

        if max_switch_count is not None and injecting.any():
            mask_list = [injecting[i].item() and len(inject_queues[i]) > 0 for i in range(cur_batch)]
            force_mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
            if force_mask.any():
                force_toks = torch.tensor([inject_queues[i].pop(0) for i in range(cur_batch) if mask_list[i]], device=device, dtype=torch.long)
                next_tokens[force_mask] = force_toks
            if injecting.any():
                done_mask = torch.tensor([injecting[i] and (len(inject_queues[i]) == 0) for i in range(cur_batch)], device=device, dtype=torch.bool)
                injecting[done_mask] = False

        cur_entropy = -(probs_original * (probs_original.clamp(min=1e-8).log())).sum(dim=-1)
        to_soft = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        to_normal = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        if step == 0:
            cur_ref_entropy = cur_entropy.clone()
        else:
            mode_stay_steps += 1
            allow_switch = (mode_stay_steps >= window_size)
            to_normal = (mode == 0) & (cur_entropy < cur_ref_entropy) & (cur_entropy < b2)
            to_soft = (mode == 1) & (cur_entropy > cur_ref_entropy) & allow_switch & (~locked_normal_mask) & (cur_entropy > b1)

            if (to_normal.any() or to_soft.any()) and step % 50 == 0:
                print(f"[LEAD-ATTEN] Step {step}: to_normal={to_normal.nonzero().squeeze(-1).tolist()}, to_soft={to_soft.nonzero().squeeze(-1).tolist()}")

            mode[to_normal] = 1
            mode[to_soft] = 0
            mode_stay_steps[to_normal | to_soft] = 0
            cur_ref_entropy[to_normal | to_soft] = cur_entropy[to_normal | to_soft]

            if to_normal.any() or to_soft.any():
                print(f"[LEAD-ATTEN] Step {step}: switch entropy={cur_entropy.mean().item():.4f}")

            if max_switch_count is not None:
                switch_count = switch_count + to_normal.long()

        is_normal = (mode == 1) | locked_normal_mask
        if math_ids_tensor is not None:
            is_math_token = (next_tokens.unsqueeze(-1) == math_ids_tensor).any(dim=-1)
            is_normal[is_math_token] = True
        is_soft = ~is_normal

        normal_emb = E[next_tokens]
        soft_emb = torch.matmul(probs_original, E)

        alpha = alpha_0 + (1 - alpha_0) * float(step) / float(max_new_tokens)
        lambda_t = max(0.0, min(1.0, a * (1.0 - alpha)))
        if step == 0:
            soft_emb = 0.9 * soft_emb + 0.1 * line_break_emb

        guided_soft_emb = soft_emb
        anchor_applied = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        if step > 0 and to_soft.any():
            attn_layers = outputs.attentions[-4:] if outputs.attentions is not None else ()
            dynamic_anchor, has_anchor = _compute_dynamic_visual_anchor(
                attn_layers=attn_layers,
                soft_emb=soft_emb,
                prompt_hidden_states=prompt_hidden_states,
                visual_token_mask=visual_token_mask,
                prompt_len=prompt_len,
                top_m=visual_anchor_top_m,
            )
            apply_anchor = to_soft & has_anchor.to(to_soft.device)
            anchor_applied = apply_anchor.to(device)
            guided_candidates = (1.0 - lambda_t) * soft_emb + lambda_t * dynamic_anchor.to(soft_emb.device)
            guided_soft_emb = torch.where(
                apply_anchor[:, None],
                guided_candidates,
                soft_emb,
            )
            print(
                f"[LEAD-ATTEN] Step {step}: visual_anchor_applied="
                f"{anchor_applied.nonzero().squeeze(-1).tolist()}"
            )

        beta = beta_0 + (1 - beta_0) * float(step) / float(max_new_tokens)

        if step % 200 == 0:
            print(
                f"[LEAD-ATTEN] Step {step}: alpha={alpha:.3f}, beta={beta:.3f}, "
                f"lambda={lambda_t:.3f}, soft_ratio={(mode==0).float().mean():.2f}"
            )

        if step > 0:
            mixed_emb = beta * guided_soft_emb + (1 - beta) * end_thinking_emb
            normal_emb = torch.where(to_normal[:, None], mixed_emb, normal_emb)
        last_emb = torch.where(is_soft[:, None], guided_soft_emb, normal_emb)

        if max_switch_count is not None and step > 0:
            trigger = (switch_count >= max_switch_count) & (switch_count <= 2 * max_switch_count) & to_normal
            if trigger.any():
                print(f"[LEAD-ATTEN] Inject convergence at step {step}, sample={trigger.nonzero().squeeze(-1).tolist()}")
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(convergence_ids)
                injecting = injecting | trigger

            trigger = (switch_count > 2 * max_switch_count) & to_normal
            if trigger.any():
                print(f"[LEAD-ATTEN] Inject termination at step {step}, sample={trigger.nonzero().squeeze(-1).tolist()}")
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(termination_ids)
                injecting = injecting | trigger
                answer_budget[trigger] = termination_max_tokens
            active = (answer_budget >= 0)
            if active.any():
                answer_budget = torch.where(active, answer_budget - 1, answer_budget)

        for bi, orig in enumerate(unfinished_idx):
            token_id = next_tokens[bi].item()
            all_generated[orig].append(token_id)
            if token_trace is not None:
                token_trace.append({
                    "step": int(step),
                    "batch_index": int(orig),
                    "token_id": int(token_id),
                    "raw_entropy": float(cur_entropy[bi].item()),
                    "filtered_entropy": float(filtered_entropy[bi].item()),
                    "selected_prob": float(probs[bi, next_tokens[bi]].item()),
                    "mode": "soft" if bool(is_soft[bi].item()) else "normal",
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "lambda_t": float(lambda_t),
                    "anchor_applied": bool(anchor_applied[bi].item()),
                })
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])

        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        if max_switch_count is not None:
            budget_done = (answer_budget == 0)
            cur_finished = cur_finished | budget_done

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]
        if len(unfinished_idx) == 0:
            break
        last_emb = last_emb[keep_idx]
        attention_mask = attention_mask[keep_idx]
        mode = mode[keep_idx]
        mode_stay_steps = mode_stay_steps[keep_idx]
        cur_ref_entropy = cur_ref_entropy[keep_idx]
        locked_normal_mask = locked_normal_mask[keep_idx]
        prompt_hidden_states = prompt_hidden_states[keep_idx]
        visual_token_mask = visual_token_mask[keep_idx]
        if hasattr(past_key_values, "batch_select_indices"):
            keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
            past_key_values.batch_select_indices(keep_idx_tensor)
        if max_switch_count is not None:
            switch_count = switch_count[keep_idx]
            injecting = injecting[keep_idx]
            inject_queues = [inject_queues[i] for i in keep_idx.tolist()]
            answer_budget = answer_budget[keep_idx]

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out
