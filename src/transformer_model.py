from src.transformer_criteria import get_config
from transformers import GPT2LMHeadModel
import torch, torch.nn as nn

model = GPT2LMHeadModel(get_config())


def efficient_attn(q, k, v, attn_mask, hsize, a, topk=32):
    # Project k and v from [B, heads, L, D] to [B, heads, proj_dim, D] using learnable matrices
    k_proj = torch.einsum('nlp,bnld->bnpd', a.E, k)
    v_proj = torch.einsum('nlp,bnld->bnpd', a.F, v)
    scores = torch.matmul(q, k_proj.transpose(-2, -1)) / (hsize ** 0.5)
    if attn_mask is not None:
        scores = scores + attn_mask
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    # Apply sparse attention by keeping only the top-k values and re-normalizing
    if topk is not None:
        top_vals, top_idx = torch.topk(attn_weights, k=topk, dim=-1)
        sparse_weights = torch.zeros_like(attn_weights)
        sparse_weights.scatter_(-1, top_idx, top_vals)
        attn_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
    out = torch.matmul(attn_weights, v_proj)
    return out, attn_weights


def patch_efficient_attention(m, proj_dim=64):
    for block in m.transformer.h:
        a = block.attn
        if not hasattr(a, "E"):
            bs = a.bias.size(-1)  # block size
            nh = a.num_heads
            a.E = nn.Parameter(torch.randn(nh, bs, proj_dim, device=a.bias.device))
            a.F = nn.Parameter(torch.randn(nh, bs, proj_dim, device=a.bias.device))
        a._attn = lambda q, k, v, attn_mask=None, head_mask=None, a=a: efficient_attn(q, k, v, attn_mask, q.size(-1), a, topk=32)

patch_efficient_attention(model)
print(model)
print("Efficient attention patched.") 