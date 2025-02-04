import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear, r=4, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(linear.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, linear.out_features) * 0.01)
        self.linear.weight.requires_grad = False  # freeze original weights
        self.bias = linear.bias  # keep bias unchanged

    def forward(self, x):
        out = self.linear(x)
        out += self.alpha * (x @ self.lora_A @ self.lora_B)
        return out


def apply_lora(model, r=4, alpha=1.0):
    # Apply LoRA adapter to transformer attention modules (c_attn) for parameter-efficient fine-tuning
    for block in model.transformer.h:
        if hasattr(block.attn, "c_attn"):
            block.attn.c_attn = LoRALinear(block.attn.c_attn, r, alpha) 