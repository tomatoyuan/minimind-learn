# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型的配置类。
    继承自 Hugging Face 的 PretrainedConfig，支持模型的配置保存、加载以及与 AutoConfig 兼容。
    该配置涵盖了标准 Transformer 参数、RoPE (旋转位置编码) 扩展设置以及 MoE (混合专家) 专用设置。
    """
    model_type = "minimind" # 模型类型标识符，用于 AutoModel 自动识别

    def __init__(
            self,
            # === 基础 Transformer 参数 ===
            dropout: float = 0.0,            # Dropout 概率，用于防止过拟合 (通常预训练为0，微调可开启)
            bos_token_id: int = 1,           # 序列开始 Token 的 ID (Begin of Sentence)
            eos_token_id: int = 2,           # 序列结束 Token 的 ID (End of Sentence)
            hidden_act: str = 'silu',        # 隐藏层激活函数，这里使用 SiLU (Swish)，是 LLaMA 等现代 LLM 的标配
            hidden_size: int = 512,          # 模型嵌入维度 (d_model)，决定了模型的宽度
            intermediate_size: int = None,   # FFN (前馈神经网络) 的中间层维度，通常是 hidden_size 的 2-4 倍(常用 8/3 倍)
            max_position_embeddings: int = 32768, # 模型能处理的最大上下文长度 (Context Window)
            num_attention_heads: int = 8,    # Query (查询) 的注意力头数
            num_hidden_layers: int = 8,      # Transformer Block 的层数 (深度)
            
            # === GQA (Grouped Query Attention) 参数 ===
            num_key_value_heads: int = 2,    # Key 和 Value 的头数。
                                             # 若此值 < num_attention_heads，则表示开启 GQA。
                                             # 作用：大幅降低推理显存占用和 KV Cache 大小，提升推理速度。
            
            vocab_size: int = 6400,          # 词表大小，决定 Embedding 层的行数
            rms_norm_eps: float = 1e-05,     # RMSNorm 的分母修正项 (epsilon)，防止除零错误
            
            # === RoPE (旋转位置编码) 相关参数 ===
            rope_theta: int = 1000000.0,     # RoPE 的基频参数。数值越大，模型对长距离依赖的衰减越慢，越适合长文本。
            inference_rope_scaling: bool = False, # 开关：是否在推理阶段开启 RoPE 缩放 (用于外推长度)
            
            # === 硬件加速 ===
            flash_attn: bool = True,         # 开关：是否使用 Flash Attention 2 加速计算 (需硬件支持)

            ####################################################
            # MOE (混合专家模型) 专属配置区域
            # 只有当 use_moe 为 True 时，以下参数才会在模型逻辑中生效
            ####################################################
            use_moe: bool = False,           # 主开关：是否启用 MoE 架构
            num_experts_per_tok: int = 2,    # Top-K：每个 Token 在推理时激活的专家数量 (通常远小于总专家数)
            n_routed_experts: int = 4,       # 路由专家 (Routed Experts) 的总数：待选专家的总池子大小
            n_shared_experts: int = 1,       # 共享专家 (Shared Experts) 的数量：
                                             # 借鉴 DeepSeek-V2/V3 设计，这些专家总是被激活，不参与路由竞争，用于捕获通用知识。
            scoring_func: str = 'softmax',   # 门控网络 (Gate) 计算路由权重的函数，通常为 softmax
            aux_loss_alpha: float = 0.1,     # 辅助损失 (Auxiliary Loss) 系数：
                                             # 用于训练时平衡各个专家的负载，防止 "专家坍塌" (Expert Collapse)。
            seq_aux: bool = True,            # 辅助损失计算方式：True 表示在序列级别统计负载，False 表示在 Batch 级别统计。
            norm_topk_prob: bool = True,     # 是否对选出的 Top-K 专家的权重进行归一化 (使其和为1)。
            **kwargs                         # 接收其他未显式定义的参数并传给父类
    ):
        super().__init__(**kwargs)
        
        # 将传入的参数绑定到实例属性上
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

        # === YaRN (Yet another RoPE extensioN) 配置逻辑 ===
        # 这是一个用于长度外推 (Extrapolation) 的技术。
        # 允许模型在推理时处理比训练长度 (original_max_position_embeddings) 更长的文本。
        # 这里设置为如果开启 inference_rope_scaling，则应用 YaRN 配置，扩展倍数为 4 倍。
        self.rope_scaling = {
            "beta_fast": 4,      # YaRN 高频衰减参数
            "beta_slow": 1,      # YaRN 低频衰减参数
            "factor": 4,         # 线性扩展因子 (Context window * 4)
            "original_max_position_embeddings": 2048, # 模型原始预训练时的最大长度
            "type": "yarn"       # 指定缩放类型为 yarn
        } if self.inference_rope_scaling else None
        
        self.flash_attn = flash_attn

        ####################################################
        # MoE 参数绑定
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # Top-K
        self.n_routed_experts = n_routed_experts        # 专家池大小
        self.n_shared_experts = n_shared_experts        # 共享专家数
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha            # 负载均衡 Loss 权重
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob



# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma (weight)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # rsqrt 是 1/sqrt(x)
        # pow(2).mean(-1) 计算 x^2 的均值
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 强制转为 float32 计算 norm 以保证数值精度，最后再转回 x 的类型 (如 float16)
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    # 1. 基础频率计算（θ）
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 2. YaRN 扩展算法（长文本外推逻辑）
    if rope_scaling is not None:
        # 获取配置参数
        original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 2048) # 
        factor = rope_scaling.get("factor", 4) # 扩展倍数
        beta_fast = rope_scaling.get("beta_fast", 4.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)

        # 仅当推理长度 end 超过训练长度 orig_max 时触发
        if end / original_max_position_embeddings > 1:
            # 寻找高频和低频的分界点 corr_dim
            # 后续会截取 freqs[0:corr_dim] 对应的频率分量（即有效低频率部分），过滤掉高频噪声或冗余的高频分量
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > original_max_position_embeddings), dim // 2)

            # 计算插值斜坡 (Ramp function)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            
            # YaRN 缩放公式
            # 对高频分量（dim 较小部分）不怎么缩放，对低频分量（dim 较大部分）进行强缩放
            scale = torch.where(torch.arange(dim // 2, device=freqs.device) < corr_dim, 
                                (beta * factor - beta + 1) / (beta * factor), 
                                1.0 / factor)
            freqs = freqs * scale # 修正频率

    # 3. 生成位置编码
    t = torch.arange(end, device=freqs.device)
    # 外积：生成 [seq_len, dim//2] 的角度矩阵 theta * position
    freqs = torch.outer(t, freqs).float()
    
    # 4. 拼接 Cos 和 Sin
    # 注意：这里拼接了两次，是为了适配下面的 rotate_half 实现
    # 形状变为 [seq_len, dim]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    '''
        !!!!!!! 此处向量的维度暂不确定，等后续debug时再确认!!!!!!

        应用旋转位置编码（RoPE）到查询向量 q 和键向量 k。
        
        参数:
        - q: 查询向量，形状为 [batch_size, seq_len, num_heads, head_dim]
        - k: 键向量，形状为 [batch_size, seq_len, num_heads, head_dim]
        - cos: 预计算的余弦频率，形状为 [seq_len, head_dim]
        - sin: 预计算的正弦频率，形状为 [seq_len, head_dim]
        - position_ids: 位置索引，形状为 [batch_size, seq_len]
        
        返回:
        - q_embed: 编码后的查询向量
        - k_embed: 编码后的键向量
    '''
    def rotate_half(x):
        '''辅助函数：将向量切分为两半，并交换顺序、取负
           [x1, x2] -> [-x2, x1]
        '''
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

    # 应用欧拉公式的实数形式
    # q_embed = q * cos + rotate_half(q) * sin
    # 张量形状变化：
    # q: [batch_size, seq_len, num_heads, head_dim]
    # k: [batch_size, seq_len, num_heads, head_dim]
    # cos/sin: [seq_len, head_dim]
    # unsqueeze_dim: 1 表示在 seq_len 维度上扩展，将 cos/sin 扩展为 [seq_len, 1, head_dim]
    # 广播机制：计算时自动将最左侧维度对齐，将 cos/sin 扩展为 [batch_size, seq_len, 1, head_dim]
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值张量的最后一个维度 n_rep 次。

    参数:
    - x: 输入张量，形状为 [batch_size, seq_len, num_heads, head_dim]
    - n_rep: 重复次数

    返回:
    - 重复后的张量，形状为 [batch_size, seq_len, num_heads, head_dim * n_rep]
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # None：等价于np.newaxis，作用是在指定位置插入一个长度为 1 的新维度（这里插入在第 3 维），下面举例示意维度变化
    # (2,10,8,64) → (2,10,8,1,64) →                 (2,10,8,2,64)                     → (2,10,8,128)
    return x[:, :, :, None, :].expand(batch_size, seq_len, num_heads, n_rep, head_dim).reshape(batch_size, seq_len, num_heads, head_dim * n_rep)

class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        self.n_local_heads = args.num_key_value_heads
        self.n_local_kv_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # hasattr(object, name) —— Python 内置函数: 判断一个对象（object）是否具有指定名称（name）的属性或方法，返回布尔值
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
    
    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor], # 修改为接收 cos 和 sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None
                ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache 实现
        if past_key_value is not None:
            # 在 seq_len 维度上拼接缓存的键值对
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len > 1:
            # 只有 “启用了 Flash Attention” 且 “序列长度有效（>1）”，才进入后续掩码处理逻辑；否则跳过（使用普通注意力的掩码逻辑）。
            if attention_mask is None or torch.all(attention_mask == 1):
                # 用户未传入任何注意力掩码（默认无掩码）或者传入的掩码是 “全 1 掩码”（所有位置都允许注意力加权，等价于无掩码）；
                # 则将 attn_mask 设置为 None（表示无掩码），并将 is_causal 设置为 True（表示使用因果掩码）。
                attn_mask, is_causal = None, True
            else:
                # 用户传入了非全 1 的attention_mask（比如某些位置是 0，需要屏蔽这些位置的注意力）。
                # torch.triu会将非上三角部分设为 0，而上三角部分保留原-inf，最终实现 “未来位置掩码为-inf，当前和过去位置为 0”。
                causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=xq.device), diagonal=1)
                # 当启用 Flash Attention 且有用户指定的非全 1 掩码时，把 “屏蔽未来位置” 的因果约束和 “屏蔽用户指定位置” 的普通约束，合并成一个 Flash Attention 能识别的高维掩码，同时禁用内置因果掩码避免重复处理。
                extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * float('-inf')
                attn_mask, is_causal = causal_mask.unsqueeze(0) + extended_mask, False

            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果掩码
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            # 用户指定的屏蔽位置，比如 padding
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(output)
        output = self.resid_dropout(output)
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
    
class MoEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        pass

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
        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedForward(config)
    
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # self attention
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        # FFN
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
    

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings,
                                                    rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: Optional[bool] = False,
                **kwargs
                ):
        batch_size, seq_len = input_ids.shape
        # 先清空一下 past_key_values 防止出错
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * self.num_hidden_layers
        # past_key_values: List of Tuples, 每个 Tuple 是每一个block层的 (key, value)
        # shape: (batch_size, past_seq_len, num_kv_heads, head_dim)
        # start_pos 对应下一个输入 token 的位置索引，之前对话的内容
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.embed_tokens(input_ids)  # [bsz, seq_len, hidden_size]
        hidden_states = self.dropout(hidden_states)

        position_embeddings = (
            # seq_len 对应当前输入的长度
            self.freqs_cos[start_pos: start_pos + seq_len],
            self.freqs_sin[start_pos: start_pos + seq_len]
        )

        presents = [] # 当前的 KV
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 逐个 block 进行运算
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        
        hidden_states = self.norm(hidden_states)

        # 计算 MoE 辅助损失
        aux_loss = sum(
            layer.mlp.aux_loss for layer in self.layers if hasattr(layer.mlp, MoEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    # 继承 PreTrainedModel 后，会自动通过 config_class 解析配置，无需手动处理超参数传递，统一管理模型结构。
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        # 1. 处理配置：若未传入config，使用默认的MiniMindConfig()
        self.config = config or MiniMindConfig()
        # 2. 调用父类PreTrainedModel的初始化（必须传入config，用于后续权重管理）
        super().__init__(self.config)
        # 3. 初始化基础模型（MiniMindModel：核心编码器/解码器，负责提取文本特征）
        self.model = MiniMindModel(self.config)
        # 4. 初始化语言建模头（lm_head）：将特征映射为词汇表概率分布
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 5. 关键设计：词嵌入层与lm_head共享权重【将hidden映射到词表大小的线性层与词嵌入层共享权重，实现高效的词汇表映射】
        #   词嵌入层：token 索引 → 嵌入向量（用共享权重的「行」）；
        #   lm_head：特征向量 → logits（用共享权重的「列」）；
        #   两者共享权重，保证「嵌入→特征→生成」的语义一致性，同时减少参数量。
        self.model.embed_tokens.weight = self.lm_head.weight
        # 6. 初始化输出容器（可选：存储模型输出的结构化对象）
        self.OUT = CausalLMOutputWithPast()

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: Optional[bool] = False,
                # logits可以存储 int 或者 tensor 类型
                # 「保留前 N 个 logits」 or 「按张量中的索引保留 logits」比如传入张量 torch.tensor([1,3,5])，就是保留第 1、3、5 个位置的 logits
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        # 目的是「筛选出需要计算 logits 的 token 位置」，避免对整个序列的 logits 做冗余计算（比如训练时只需要预测「下一个 token」，只需保留最后一个 token 的 logits）
        # 情况 1：logits_to_keep 是 int 类型（如 3）：
        #   - slice(-3, None) 等价于「从序列倒数第 3 个元素到末尾」（Python 的切片语法）；
        #   - 比如序列长度为 5，slice(-3, None) 对应索引 [2,3,4]（保留最后 3 个 token）。
        # 情况 2：logits_to_keep 是 torch.Tensor 类型（如 torch.tensor([1,3,5])）：
        #   - 直接将张量作为索引（需保证张量是整数类型，且索引在序列长度范围内）；
        #   - 用于精准保留特定位置的 logits（如某些任务需要重点关注中间几个 token 的预测）。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 只计算需要的 logits
        # 输入：h[:, slice_indices, :] → 筛选后的特征，形状为：
        #   - 若 logits_to_keep 是 int=3：[batch_size, 3, hidden_size]；
        #   - 若 logits_to_keep 是张量 [1,3,5]：[batch_size, 3, hidden_size]（3 是索引个数）；
        # 关键优化：只对「需要的 token 特征」做线性映射，避免对整个序列（比如长度 1024）的所有 token 计算 logits，
        # 节省计算资源（尤其是生成时，只需计算最后一个 token 的 logits）。
        logits = self.lm_head(h[:, slice_indices, :])  
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT