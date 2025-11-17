# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class Mock_RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    # _norm 
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    # forward
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), rope_scaling.get("beta_slow", 1.0)
        )
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            # λ = (β·α - β + 1)/(β·α) YaRN标准公式
            scale = torch.where(torch.arange(dim // 2, device=freqs.device) < corr_dim, (beta * factor - beta + 1) / (beta * factor), 1.0 / factor)
            freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def Mock_precompute_freqs_cis(
        dim: int, 
        end: int = int(32 * 1024), 
        rope_base: float = 1e6, 
        rope_scaling: Optional[dict] = None):
    # 计算频率的公式为 1 / (rope_base ** (2i / dim))，其中i是位置索引，dim是维度
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
                rope_scaling.get("original_max_position_embeddings", 2048), 
                rope_scaling.get("factor", 4),
                rope_scaling.get("beta_fast", 4.0), 
                rope_scaling.get("beta_slow", 1.0)
        )
        # 如果推理长度真的大于原始长度，则需要进行长度外推，其实这里的逻辑有些冗余
        if end / orig_max > 1.0:
        # 计算corr_dim 的公式为 2 * math.pi / freqs[i] > orig_max，其中i是位置索引
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
        # 计算power 的公式为 i / max(dim // 2 - 1, 1)，其中i是位置索引
            power = torch.arange(0, dim // 2, device = freqs.device).float() / max(dim // 2 - 1, 1)
        # 计算beta 的公式为 beta_slow + (beta_fast - beta_slow) * power
            beta = beta_slow + (beta_fast - beta_slow) * power
        # 计算scale 的公式为 (beta * factor - beta + 1) / (beta * factor)，其中beta是beta，factor是factor
            scale = torch.where(
                torch.arange(dim // 2, device = freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor
            )
        # 应用scale
            freqs = freqs * scale

    # 生成位置索引，与频率相乘
    t = torch.arange(end, device = freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 返回一个cos和一个sin的矩阵，分别对应频率的cos和sin值
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1)
    return freqs_cos, freqs_sin

def Mock_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # [a, b] -> [-b, a]
    def rotate_half(x):
        # 将x的右半部分和左半部分交换
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

def Mock_reqeat_kv(x : torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].
        expand(bs, slen, num_key_value_heads, n_rep, head_dim).
        reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class Mock_Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()

        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0
        "num_attention_heads must be divisible by num_key_value_heads"

        # 多头注意力机制的参数
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # 输入投影层，将输入映射到多头注意力机制的维度
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出投影层，将多头注意力输出映射回原始维度
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
    
    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            attention_mask: Optional[torch.Tensor] = None
        ):
        """
        Attention Forward Pass:
        
        ┌─────────────────────────────────────────────┐
        │  Input: x                                   │
        └────────────────┬────────────────────────────┘
                         │
                         ▼
              ┌──────────────┐
              │   q_proj     │
              │   k_proj     │
              │   v_proj     │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   view       │
              │  (multi-head)│
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   ROPE       │
              │(position emb)│
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  KV Cache    │
              │  (if exists) │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ repeat_kv    │
              │ transpose    │
              └──────┬───────┘
                     │
                     ├──────────────┐
                     │              │
                     ▼              ▼
            ┌─────────────┐   ┌─────────────┐
            │Flash Attn   │   │Slow Attn    │
            │(SDPA)       │   │(QK^T/sqrt)  │
            └──────┬──────┘   └──────┬──────┘
                   │                 │
                   │                 ▼
                   │         ┌─────────────┐
                   │         │Causal Mask  │
                   │         └──────┬──────┘
                   │                │
                   │                ▼
                   │         ┌─────────────┐
                   │         │Attn Mask    │
                   │         │(if provided)│
                   │         └──────┬──────┘
                   │                │
                   │                ▼
                   │         ┌─────────────┐
                   │         │Softmax      │
                   │         │Dropout      │
                   │         └──────┬──────┘
                   │                │
                   │                ▼
                   │         ┌─────────────┐
                   │         │scores @ V   │
                   │         └──────┬──────┘
                   │                │
                   └────────┬───────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ transpose    │
                    │ reshape      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   o_proj     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  dropout     │
                    └──────┬───────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────┐
        │  Output: (output, past_kv)                  │
        └─────────────────────────────────────────────┘
        """
        bsz, seq_len, _ = x.shape
        # 投影，计算q, k, v
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 把输入拆分成多个头，用view
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # q和k 使用 ROPE

        cos, sin = position_embeddings
        xq, xk = Mock_apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # transpose(1, 2): 将形状从 (bsz, seq_len, n_heads, head_dim) 转换为 (bsz, n_heads, seq_len, head_dim)
        # 目的是将多头注意力的维度顺序调整为适合attention计算的格式（batch, heads, seq_len, head_dim）
        xq, xk, xv = (
            xq.transpose(1, 2),  # (bsz, seq_len, n_local_heads, head_dim) -> (bsz, n_local_heads, seq_len, head_dim)
            Mock_reqeat_kv(xk, self.n_rep).transpose(1, 2),  # (bsz, seq_len, n_local_kv_heads, head_dim) -> (bsz, n_local_heads, seq_len, head_dim)
            Mock_reqeat_kv(xv, self.n_rep).transpose(1, 2)   # (bsz, seq_len, n_local_kv_heads, head_dim) -> (bsz, n_local_heads, seq_len, head_dim)
        )
        # 进行attention计算
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )

            # 使用flash attention
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 使用 slow attention
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0) # scores+mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask # scores+mask
            # softmax
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # transpose(1, 2): 将形状从 (bsz, n_heads, seq_len, head_dim) 转换为 (bsz, seq_len, n_heads, head_dim)
        # reshape: 将形状从 (bsz, seq_len, n_heads, head_dim) 转换为 (bsz, seq_len, n_heads * head_dim)
        # 目的是将多头注意力的输出重新拼接成原始格式，准备进行输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # (bsz, n_local_heads, seq_len, head_dim) -> (bsz, seq_len, n_local_heads, head_dim) -> (bsz, seq_len, hidden_size)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv



class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class Mock_FeedForward(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 升维
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        # 降维
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        # 门控
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        # dropout
        self.dropout = nn.Dropout(args.dropout)
        # 激活函数
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        """
        FeedForward Forward Pass (SwiGLU Activation):
        
        ┌─────────────────────────────────────────────────────────────┐
        │  Input: x                                                   │
        │  Shape: (bsz, seq_len, hidden_size)                         │
        └────────────────┬────────────────────────────────────────────┘
                         │
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
        ┌─────────────┐         ┌─────────────┐
        │ gate_proj   │         │  up_proj    │
        │ (Gate Proj) │         │  (Up Proj)  │
        └──────┬──────┘         └──────┬──────┘
               │                       │
               │ (bsz, seq_len, inter) │ (bsz, seq_len, inter)
               │                       │
               ▼                       │
        ┌─────────────┐                │
        │   act_fn    │                │
        │ (SiLU Act)  │                │
        └──────┬──────┘                │
               │                       │
               └───────┬───────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Element-wise (×) │
              │  gate × up       │
              └────────┬─────────┘
                       │
                       │ (bsz, seq_len, inter)
                       ▼
              ┌─────────────┐
              │  down_proj  │
              │ (Down Proj) │
              └──────┬──────┘
                     │
                     │ (bsz, seq_len, hidden_size)
                     ▼
              ┌─────────────┐
              │   dropout   │
              │(Regularize) │
              └──────┬──────┘
                     │
                     ▼
        ┌─────────────────────────────────────────────┐
        │  Output: (bsz, seq_len, hidden_size)        │
        └─────────────────────────────────────────────┘
        
        Formula: output = dropout(down_proj(act_fn(gate_proj(x)) * up_proj(x)))
        """
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MoEGate(nn.Module):
    """
    MoE (Mixture of Experts) Gate: 用于为每个token选择最合适的专家
    门控机制通过计算每个token对所有专家的评分，选择top-k个专家
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.config = config  # 保存配置对象
        
        # 每个token选择的专家数量（top-k）
        self.top_k = config.num_experts_per_tok
        
        # 路由专家的总数（不包括共享专家）
        self.n_routed_experts = config.n_routed_experts

        # 评分函数类型，目前只支持'softmax'
        self.scoring_func = config.scoring_func
        
        # 辅助损失的权重系数，用于平衡负载均衡损失
        self.alpha = config.aux_loss_alpha
        
        # 是否在序列级别计算辅助损失（True）还是在token级别（False）
        self.seq_aux = config.seq_aux

        # 是否对top-k概率进行归一化
        self.norm_topk_prob = config.norm_topk_prob
        
        # 门控网络的输入维度，等于隐藏层维度
        self.gating_dim = config.hidden_size
        
        # 门控权重矩阵: (n_routed_experts, gating_dim)
        # 每一行对应一个专家的权重向量，用于计算token对该专家的评分
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        
        # 初始化权重参数
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        使用Kaiming均匀分布初始化权重
        a=math.sqrt(5) 是ReLU激活函数的推荐参数
        虽然这里用的是线性层，但使用Kaiming初始化仍然有效
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        前向传播：为每个token选择top-k个专家
        
        Args:
            hidden_states: (bsz, seq_len, hidden_size) 输入隐藏状态
            
        Returns:
            topk_idx: (bsz*seq_len, top_k) 每个token选择的top-k专家索引
            topk_weight: (bsz*seq_len, top_k) 每个token对选择的专家的权重
            aux_loss: 辅助损失（负载均衡损失）
        """
        # 获取输入张量的形状
        bsz, seq_len, h = hidden_states.shape
        
        # 将输入重塑为 (bsz*seq_len, hidden_size)
        # 这样每个token都被视为独立的样本，便于批量处理
        hidden_states = hidden_states.view(-1, h)
        
        # 计算每个token对所有专家的原始评分（logits）
        # F.linear(x, weight, bias) = x @ weight.T + bias
        # 结果形状: (bsz*seq_len, n_routed_experts)
        # 每一行表示一个token对所有专家的评分
        logits = F.linear(hidden_states, self.weight, None)
        
        # 将logits转换为概率分布
        if self.scoring_func == 'softmax':
            # 对每个token的专家评分进行softmax归一化
            # 结果形状: (bsz*seq_len, n_routed_experts)
            # 每一行的和等于1，表示该token对各个专家的选择概率
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 为每个token选择top-k个专家
        # topk_idx: (bsz*seq_len, top_k) 选择的专家索引
        # topk_weight: (bsz*seq_len, top_k) 对应的权重（概率）
        # sorted=False 表示不按权重排序，保持原始顺序
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果top_k > 1 且需要归一化top-k概率
        if self.top_k > 1 and self.norm_topk_prob:
            # 计算top-k权重的和（每个token的top-k权重之和）
            # keepdim=True 保持维度，便于广播除法
            # + 1e-20 防止除零
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            
            # 将top-k权重归一化，使得每个token的top-k权重之和为1
            # 这样确保每个token分配给top-k专家的总权重为1
            topk_weight = topk_weight / denominator

        # 计算辅助损失（负载均衡损失）
        # 只在训练时且alpha > 0时计算
        if self.training and self.alpha > 0.0:
            # 保存完整的scores用于辅助损失计算
            scores_for_aux = scores
            
            # top-k的值
            aux_topk = self.top_k
            
            # 将topk_idx重塑为 (bsz, seq_len*top_k)
            # 便于后续计算每个batch中每个专家被选择的次数
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            # 根据配置选择不同的辅助损失计算方式
            if self.seq_aux:
                # 序列级别的辅助损失
                # 将scores重塑为 (bsz, seq_len, n_routed_experts)
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                
                # 初始化每个batch中每个专家被选择的计数
                # ce: (bsz, n_routed_experts) 表示每个batch中每个专家被选择的次数
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                
                # scatter_add_: 根据索引累加
                # 将每个token选择的top-k专家索引对应的位置加1
                # 结果: ce[i, expert_idx] += 1 对于每个被选择的专家
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device))
                
                # 归一化：将计数除以期望值
                # 期望值 = seq_len * aux_topk / n_routed_experts
                # 理想情况下，每个专家应该被选择 seq_len * aux_topk / n_routed_experts 次
                ce.div_(seq_len * aux_topk / self.n_routed_experts)
                
                # 计算辅助损失
                # scores_for_seq_aux.mean(dim=1): (bsz, n_routed_experts) 每个batch中每个专家的平均评分
                # ce: (bsz, n_routed_experts) 归一化后的选择计数
                # 两者相乘后求和再平均，最后乘以alpha权重
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Token级别的辅助损失（全局平均）
                # 将topk_idx展平为一维: (bsz*seq_len*top_k,)
                # 然后转换为one-hot编码: (bsz*seq_len*top_k, n_routed_experts)
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                
                # 计算每个专家被选择的平均频率
                # ce: (n_routed_experts,) 每个专家被选择的平均频率
                ce = mask_ce.float().mean(0)
                
                # 计算每个专家的平均评分
                # Pi: (n_routed_experts,) 每个专家的平均评分
                Pi = scores_for_aux.mean(0)
                
                # 计算负载因子
                # fi = ce * n_routed_experts
                # 理想情况下，每个专家应该被选择 1/n_routed_experts 的比例
                # 所以 fi 应该接近1
                fi = ce * self.n_routed_experts
                
                # 计算辅助损失: sum(Pi * fi) * alpha
                # 鼓励负载均衡：如果某个专家被过度使用（fi > 1），会增加损失
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 推理时或alpha=0时不计算辅助损失
            aux_loss = 0
        
        # 返回选择的专家索引、权重和辅助损失
        return topk_idx, topk_weight, aux_loss

class Mock_MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k = self.top_k, dim = -1, sorted = False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores

            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                )
                ce.div_(seq_len * aux_topk / self.n_routed_experts)

                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1),
                                    num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    """
    MoE (Mixture of Experts) FeedForward: 混合专家前馈网络
    通过门控机制为每个token选择top-k个专家，并将专家输出加权求和
    包含路由专家（routed experts）和可选的共享专家（shared experts）
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.config = config  # 保存配置对象
        
        # 创建路由专家列表
        # 每个专家都是一个独立的FeedForward网络
        # 数量由config.n_routed_experts决定
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        
        # 创建门控网络，用于为每个token选择专家
        self.gate = MoEGate(config)
        
        # 如果配置了共享专家，创建共享专家列表
        # 共享专家对所有token都进行处理，不经过门控选择
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        前向传播：通过门控机制选择专家并计算输出
        
        Args:
            x: (bsz, seq_len, hidden_size) 输入隐藏状态
            
        Returns:
            y: (bsz, seq_len, hidden_size) 输出隐藏状态
        """
        # 保存原始输入，用于共享专家的残差连接
        identity = x
        
        # 保存原始形状，后续需要恢复
        orig_shape = x.shape
        
        # 获取batch size和序列长度
        bsz, seq_len, _ = x.shape
        
        # 使用门控机制为每个token选择top-k个专家
        # topk_idx: (bsz*seq_len, top_k) 选择的专家索引
        # topk_weight: (bsz*seq_len, top_k) 对应的权重
        # aux_loss: 辅助损失（负载均衡损失）
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 将输入重塑为 (bsz*seq_len, hidden_size)
        # 每个token被视为独立样本
        x = x.view(-1, x.shape[-1])
        
        # 将topk_idx展平为一维: (bsz*seq_len*top_k,)
        # 便于后续索引操作
        flat_topk_idx = topk_idx.view(-1)
        
        # 训练和推理使用不同的实现策略
        if self.training:
            # ========== 训练模式：简单但低效的实现 ==========
            # 将每个token复制top_k次，因为每个token需要被top_k个专家处理
            # 结果形状: (bsz*seq_len*top_k, hidden_size)
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            
            # 创建输出张量，形状与x相同
            y = torch.empty_like(x, dtype=torch.float16)
            
            # 遍历每个专家，处理分配给它的token
            for i, expert in enumerate(self.experts):
                # 找到分配给当前专家i的所有token索引
                # flat_topk_idx == i 返回布尔掩码
                # x[flat_topk_idx == i] 获取这些token的输入
                # expert(...) 通过专家网络处理
                # 将结果写入y的对应位置
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            
            # 加权求和：将每个token的top_k个专家输出按权重加权求和
            # y.view(*topk_weight.shape, -1): (bsz*seq_len, top_k, hidden_size)
            # topk_weight.unsqueeze(-1): (bsz*seq_len, top_k, 1) 用于广播
            # 相乘后按dim=1求和: (bsz*seq_len, hidden_size)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            
            # 恢复原始形状: (bsz, seq_len, hidden_size)
            y = y.view(*orig_shape)
        else:
            # ========== 推理模式：优化的批量处理实现 ==========
            # 使用moe_infer方法进行批量处理，提高效率
            # flat_topk_idx: (bsz*seq_len*top_k,) 展平的专家索引
            # topk_weight.view(-1, 1): (bsz*seq_len*top_k, 1) 展平的权重
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 如果配置了共享专家，将共享专家的输出加到结果中
        # 共享专家对所有token都进行处理，不经过门控选择
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                # 共享专家使用原始输入identity，而不是处理后的x
                y = y + expert(identity)
        
        # 保存辅助损失，供模型训练时使用
        self.aux_loss = aux_loss
        
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理时的优化实现：批量处理每个专家的所有token
        
        核心思想：将token按专家分组，批量处理每个专家的所有token，然后加权累加
        
        Args:
            x: (bsz*seq_len, hidden_size) 输入token
            flat_expert_indices: (bsz*seq_len*top_k,) 展平的专家索引
            flat_expert_weights: (bsz*seq_len*top_k, 1) 展平的权重
            
        Returns:
            expert_cache: (bsz*seq_len, hidden_size) 输出token
        """
        # 初始化输出缓存，形状与输入x相同
        expert_cache = torch.zeros_like(x)
        
        # 对专家索引进行排序，使得相同专家的token聚集在一起
        # idxs: (bsz*seq_len*top_k,) 排序后的索引
        # 例如：如果flat_expert_indices = [2, 0, 1, 0, 2, 1]
        # 排序后：idxs = [1, 3, 2, 5, 0, 4]（索引1和3对应专家0，索引2和5对应专家1，...）
        idxs = flat_expert_indices.argsort()
        
        # 统计每个专家处理的token数量（累积和）
        # bincount(): 统计每个专家出现的次数
        # cumsum(0): 计算累积和，得到每个专家在排序后的数组中的结束位置
        # 例如：如果专家0有2个token，专家1有2个token，专家2有2个token
        # tokens_per_expert = [2, 4, 6] 表示专家0在位置2结束，专家1在位置4结束，专家2在位置6结束
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        
        # 计算每个token在原始输入x中的索引
        # idxs // num_experts_per_tok: 因为每个token被top_k个专家处理，需要除以top_k得到原始token索引
        # 例如：如果top_k=2，idxs = [1, 3, 2, 5, 0, 4]
        # token_idxs = [0, 1, 1, 2, 0, 2]（索引0和4对应token 0，索引1和2对应token 1，...）
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 遍历每个专家，批量处理分配给它的所有token
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前专家在排序数组中的起始位置
            # 第一个专家的起始位置是0，后续专家的起始位置是前一个专家的结束位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            
            # 如果当前专家没有处理的token，跳过
            if start_idx == end_idx:
                continue
            
            # 获取当前专家网络
            expert = self.experts[i]
            
            # 获取分配给当前专家的所有token在原始输入x中的索引
            # exp_token_idx: 当前专家需要处理的token索引列表
            exp_token_idx = token_idxs[start_idx:end_idx]
            
            # 从输入x中提取这些token
            expert_tokens = x[exp_token_idx]
            
            # 通过专家网络处理这些token
            # 转换为expert_cache的数据类型以保持一致
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            
            # 将专家输出按权重缩放
            # idxs[start_idx:end_idx]: 当前专家在排序数组中的索引
            # flat_expert_weights[idxs[...]]: 对应的权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 将加权后的输出累加到expert_cache中
            # scatter_add_: 根据token索引将输出累加到对应位置
            # exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]): 扩展索引以匹配hidden_size维度
            # 如果同一个token被多个专家处理，它们的输出会被累加
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class Mock_MoEfeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        self.experts = nn.ModuleList([
            Mock_FeedForward(config) for _ in range(config.n_routed_experts)
        ])

        self.gate = Mock_MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Mock_FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])
        
    def forward(self, x):
        """
        MoE FeedForward Forward Pass:
        
        ┌─────────────────────────────────────────────────────────────┐
        │  Input: x                                                   │
        │  Shape: (bsz, seq_len, hidden_size)                         │
        └────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Save Identity    │
              │                  │
              │ Save orig_shape  │
              │                  │
              └─────────┬────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   Gate           │
              │ (MoEGate)        │
              │ select top-k     │
              └─────────┬────────┘
                        │
                        │ topk_idx: (bsz*seq_len, top_k)
                        │ topk_weight: (bsz*seq_len, top_k)
                        │ aux_loss: scalar
                        │
                        ▼
              ┌──────────────────┐
              │   view(-1, h)    │
              │  flatten tokens  │
              │                  │
              └─────────┬────────┘
                        │
                        │ (bsz*seq_len, hidden_size)
                        │
                        ├──────────────────────────┐
                        │                          │
                        ▼                          ▼
            ┌──────────────────────┐   ┌──────────────────────┐
            │Training Mode         │   │Inference Mode        │
            │                      │   │                      │
            └──────────┬───────────┘   └──────────┬───────────┘
                       │                          │
                       ▼                          ▼
            ┌──────────────────────┐   ┌──────────────────────┐
            │repeat_interleave     │   │argsort indices       │
            │(top_k times)         │   │group by expert       │
            └──────────┬───────────┘   └──────────┬───────────┘
                       │                          │
                       │ (bsz*seq_len*top_k, h)   │
                       │                          │
                       ▼                          ▼
            ┌──────────────────────┐   ┌──────────────────────┐
            │for each expert:      │   │for each expert:      │
            │  get tokens          │   │  get tokens          │
            │  expert(tokens)      │   │  batch process       │
            └──────────┬───────────┘   └──────────┬───────────┘
                       │                          │
                       │                          │
                       ▼                          │
            ┌──────────────────────┐              │
            │view & reshape        │              │
            │(bsz*seq_len,         │              │
            │ top_k, hidden_size)  │              │
            └──────────┬───────────┘              │
                       │                          │
                       ▼                          │
            ┌──────────────────────┐              │
            │weighted sum          │              │
            │(topk_weight * out)   │              │
            │sum(dim=1)            │              │
            └──────────┬───────────┘              │
                       │                          │
                       │ (bsz*seq_len, h)         │
                       │                          │
                       └──────────┬───────────────┘
                                  │
                                  ▼
                          ┌────────────────┐  
                          │view(orig_shape)│
                          │                │
                          │reshape back    │ 
                          │                │
                          └─────────┬──────┘
                                 │
                                 │ (bsz, seq_len, hidden_size)
                                 │
                                 ├──────────────┐
                                 │              │
                                 ▼              │
                      ┌──────────────┐          │
                      │Shared Experts│          │
                      │(if exists)   │          │
                      │expert(identity)│        │
                      └──────┬───────┘          │
                             │                  │
                             │ y = y + expert_out
                             │                  │
                             └────────┬─────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Output: y (bsz, seq_len, hidden_size)                      │
        │         aux_loss: scalar                                    │
        └─────────────────────────────────────────────────────────────┘
        """
        identity = x
        orig_shape = x.shape

        bsz, seq_len, _ = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)

        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)

            y = torch.empty_like(x, dtype=torch.float16)

            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)

            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            
            y = y.view(*orig_shape)

        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)


        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)

        idxs = flat_expert_indices.argsort()

        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        token_idxs = idxs // self.config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]

            if start_idx == end_idx:
                continue

            expert = self.experts[i]
            expert_tokens_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[expert_tokens_idx]

            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, expert_tokens_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache
            

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class Mock_MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Mock_Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = Mock_RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Mock_RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Mock_FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        Transformer Block Forward Pass:
        
        ┌─────────────────────────────────────────────┐
        │  Input: hidden_states                       │
        └────────────────┬────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Save Residual   │
              └──────────┬───────┘
                         │
                         ├──────────────┐
                         │              │
                         ▼              │
              ┌──────────────────┐      │
              │ input_layernorm  │      │
              └──────────┬───────┘      │
                         │              │
                         ▼              │
              ┌──────────────────┐      │
              │   self_attn      │      │
              └──────────┬───────┘      │
                         │              │
                         ▼              │
              ┌──────────────────┐      │
              │  Residual Add (+)│◄─────┘
              └──────────┬───────┘
                         │
                         ▼
              ┌──────────────────┐
              │post_attention_   │
              │  layernorm       │
              └──────────┬───────┘
                         │
                         ├──────────────┐
                         │              │
                         ▼              │
              ┌──────────────────┐      │
              │ FeedForward (mlp)│      │
              └──────────┬───────┘      │
                         │              │
                         ▼              │
              ┌──────────────────┐      │
              │  Residual Add (+)│◄─────┘
              └──────────┬───────┘
                         │
                         ▼
        ┌─────────────────────────────────────────────┐
        │  Output: hidden_states, present_key_value   │
        └─────────────────────────────────────────────┘
        """
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), 
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss

class Mock_MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([Mock_MiniMindBlock(l, config) for l in range(config.num_hidden_layers)])
        self.norm = Mock_RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )

        self.register_buffer("freqs_cos", self.freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", self.freqs_sin, persistent=False)


        def forward(
            self, 
            input_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
            use_cache: bool,
            **kwargs
        ):
            batch_size, seq_length = input_ids.shape
            if hasattr(past_key_values, 'layers'): past_key_values = None

            past_key_values = past_key_values or [None] * len(self.layers)

            start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

            hidden_states = self.dropout(self.embed_tokens(input_ids))

            position_embeddings = (
                self.freqs_cos[start_pos:start_pos + seq_length],
                self.freqs_sin[start_pos:start_pos + seq_length]
            )

            presents = []
            for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
                hidden_states, present = layer(
                    hidden_states,
                    position_embeddings,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    attention_mask=attention_mask
                )
                presents.append(present)
            hidden_states = self.norm(hidden_states)

            aux_loss = sum(
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(layer.mlp, MOEFeedForward)
            )
            return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

class Mock_MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)

        self.model = Mock_MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 将lm_head的权重赋值给model的embed_tokens的权重
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(
            self, 
            input_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
            use_cache: bool,
            logits_to_keep: Union[int, torch.Tensor],
            **args
        ):
        hidden_states, past_kvs, aux_loss = self.model(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
            **args
        )
        
        slice_indices = (
            slice(-logits_to_keep, None) 
            if isinstance(logits_to_keep, int) 
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT