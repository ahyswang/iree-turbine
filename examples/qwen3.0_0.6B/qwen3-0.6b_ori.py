import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

class EMBD(nn.Module):
    def __init__(self, vocab_size = 151936, embd_size = 1024):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embd_size)

    def forward(self, input_ids):
        return self.embedding(input_ids)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(rms + self.eps)
        return x * rms * self.weight


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size = 1024,
        head_dim: int = 128,
        n_heads: int = 16,
        intermediate_size: int = 3072,
        max_len: int = 4096,
        rope_base: float = 1e6,
        n_kv_heads: int = 8,   # Qwen3-0.6B: 16 Q heads, 8 KV heads
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group = n_heads // n_kv_heads  # = 2
        self.head_dim = head_dim      # = 128
        self.hidden_size = hidden_size

        half_dim = self.head_dim // 2

        cos, sin = self.build_rope_tables(max_len, half_dim, rope_base)
        rope_a = torch.cat([cos, sin], dim=-1)  # [max_len, head_dim]
        rope_b = torch.cat([-sin, cos], dim=-1)

        self.rope_embed_a = nn.Embedding(max_len, self.head_dim)
        self.rope_embed_b = nn.Embedding(max_len, self.head_dim)
        with torch.no_grad():
            self.rope_embed_a.weight.copy_(rope_a)
            self.rope_embed_b.weight.copy_(rope_b)
        self.rope_embed_a.weight.requires_grad = False
        self.rope_embed_b.weight.requires_grad = False

        # block 级 pre-norm / post-norm：RMSNorm(dim)
        self.ln1 = RMSNorm(hidden_size)
        self.ln2 = RMSNorm(hidden_size)

        self.q_proj = nn.Linear(hidden_size, n_heads * self.head_dim, bias=False)         # [2048,1024]
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * self.head_dim, bias=False)      # [1024,1024]
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * self.head_dim, bias=False)      # [1024,1024]
        self.out_proj = nn.Linear(n_heads * self.head_dim, hidden_size, bias=False)       # [1024,2048]

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    # ---------- RoPE ----------
    @staticmethod
    def build_rope_tables(max_len, half_dim, base=1e6):
        # 和你之前的实现基本一样，只是 base 换掉
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)       # [L,1]
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        freqs = position * inv_freq.unsqueeze(0)                                 # [L,D/2]
        return freqs.cos(), freqs.sin()

    def apply_rope(self, x, position_ids):
        rope_a = self.rope_embed_a(position_ids).unsqueeze(1)  # [B,1,S,D]
        rope_b = self.rope_embed_b(position_ids).unsqueeze(1)  # [B,1,S,D]

        x0 = x[..., : self.head_dim // 2]
        x1 = x[..., self.head_dim // 2 :]
        x_flip = torch.cat([-x1, x0], dim=-1)

        return x * rope_a + x_flip * rope_b

    def forward(self, embd, mask, position_ids, cache_k=None, cache_v=None):
        B, S, D = embd.shape
        H = self.n_heads
        KV_H = self.n_kv_heads
        HD = self.head_dim
        G = self.group

        x = self.ln1(embd)   # [B,S,D]

        # ----- Q,K,V projection -----
        q = self.q_proj(x).view(B, S, H, HD).transpose(1, 2)        # [B,H,S,HD]
        k = self.k_proj(x).view(B, S, KV_H, HD).transpose(1, 2)     # [B,KV_H,S,HD]
        v = self.v_proj(x).view(B, S, KV_H, HD).transpose(1, 2)     # [B,KV_H,S,HD]

        # QK-Norm
        q = self.q_norm(q)   # RMSNorm over last dim
        k = self.k_norm(k)

        q = self.apply_rope(q, position_ids)
        k = self.apply_rope(k, position_ids)

        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=2)   # 在 seq 维拼接
            v = torch.cat([cache_v, v], dim=2)
        updated_k, updated_v = k, v   # 作为下一次 step 的 cache

        # GQA：repeat KV
        if G > 1:
            k_attn = k.repeat_interleave(G, dim=1)   # [B,H,S_total,HD]
            v_attn = v.repeat_interleave(G, dim=1)
        else:
            k_attn = k
            v_attn = v

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k_attn.transpose(-2, -1)) / math.sqrt(HD)  # [B,H,S,S_total]
        if mask is not None:
            attn_scores = attn_scores + mask
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v_attn)  # [B,H,S,HD]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H * HD)  # [B,S,2048]
        o1 = self.out_proj(attn_out)  # [B,S,1024]
        o2 = embd + o1             

        x_norm = self.ln2(o2)
        gate = self.gate_proj(x_norm)
        up = self.up_proj(x_norm)
        ff = self.down_proj(F.silu(gate) * up)
        out = o2 + ff         

        return out, updated_k, updated_v


class LMHead(nn.Module):
    def __init__(self, hidden_size = 1024, vocab_size = 151936):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_state):
        out = self.proj(hidden_state)
        return out
    
class Qwen3Model(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        embd_size: int = 1024,
        hidden_size: int = 1024,
        head_dim: int = 128,
        n_heads: int = 16,
        intermediate_size: int = 3072,
        n_kv_heads: int = 8,
        layers: int = 28,
        max_len: int = 4096,
        rope_base: float = 1e6,
    ):
        super().__init__()

        self.emb = EMBD(
            vocab_size=vocab_size,
            embd_size=embd_size,
        )

        self.layers = nn.ModuleList([
            Transformer(
                hidden_size = hidden_size,
                head_dim = head_dim,
                n_heads = n_heads,
                intermediate_size = intermediate_size,
                max_len = max_len,
                rope_base = rope_base,
                n_kv_heads = n_kv_heads,
            )
            for _ in range(layers)
        ])

        self.lm_head = LMHead(
            hidden_size = hidden_size,
            vocab_size = vocab_size
        )


    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        cache_k=None,
        cache_v=None
    ):
        assert isinstance(input_ids, torch.Tensor), f"Expected Tensor, got {type(input_ids)}"
        hidden_states = self.emb(input_ids)           # [B, S, D]

        for layer in self.layers:
            hidden_states, cache_k, cache_v = layer(
                hidden_states,
                attention_mask,
                position_ids,
                cache_k,
                cache_v,
            )

        logits = self.lm_head(hidden_states)          # [B,S,V]

        return logits, cache_k, cache_v


def export_transformer_block():
    model = Qwen3Model(
        vocab_size=151936,
        embd_size=1024,
        hidden_size=1024,
        head_dim=128,
        n_heads=16,
        intermediate_size=3072,
        n_kv_heads=8,
        layers=1,
        max_len=4096,
    )
    model.eval()
    

    hf_state = load_file("/workspace/yswang26/wqzhao9/qwen3-0.6b/model.safetensors")   # ← 你提前下载好的文件
    new_sd = {}

    # ================================
    # 🔥 只 remap 第0层 attention + mlp
    # ================================
    rename_map = {
        # attention
        "model.layers.0.self_attn.q_proj.weight":     "layers.0.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight":     "layers.0.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight":     "layers.0.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight":     "layers.0.out_proj.weight",

        "model.layers.0.self_attn.q_norm.weight":     "layers.0.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight":     "layers.0.k_norm.weight",

        # MLP
        "model.layers.0.mlp.up_proj.weight":          "layers.0.up_proj.weight",
        "model.layers.0.mlp.gate_proj.weight":        "layers.0.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight":        "layers.0.down_proj.weight",

        # norms
        "model.layers.0.input_layernorm.weight":      "layers.0.ln1.weight",
        "model.layers.0.post_attention_layernorm.weight":"layers.0.ln2.weight",
    }
    
    for hf_k, local_k in rename_map.items():
        if hf_k in hf_state:
            print(hf_k)
            new_sd[local_k] = hf_state[hf_k].to(torch.float16)
        else:
            print("missing HF key:", hf_k)

    # ================================
    # Load mapped weights
    # ================================
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    print(">>> weights loaded for layer 0")

    model.half()

    # 输入 dummy 数据
    B = 1
    S = 256
    MAX_CACHE = 256
    KV = 8
    HD = 128

    input_ids = torch.randint(0, 151936, (B, S), dtype=torch.int64)

    position_ids = torch.arange(S).unsqueeze(0)

    attention_mask = torch.zeros(
        B, 1, S, S + MAX_CACHE,
        dtype=torch.float16,            # <<< 必须 FP16
    )

    cache_k = torch.randn(B, KV, MAX_CACHE, HD, dtype=torch.float16)
    cache_v = torch.randn(B, KV, MAX_CACHE, HD, dtype=torch.float16)

    # 导出 ONNX
    torch.onnx.export(
        model,
        (input_ids, position_ids, attention_mask, cache_k, cache_v),
        "qwen3-0.6b.onnx",
        input_names=["input_ids", "position_ids", "attention_mask", "cache_k", "cache_v"],
        output_names=["hidden_state", "updated_k", "updated_v"],
        opset_version=13,
    )

export_transformer_block()