from transformers import GPT2LMHeadModel
from src.moe import SelectiveMoE

class MoEGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.moe = SelectiveMoE(config.n_embd, config.n_embd, num_experts=2)

    def forward(self, *args, **kwargs):
        outputs = self.transformer(*args, **kwargs)
        hidden = outputs[0] if isinstance(outputs, (list, tuple)) else outputs.last_hidden_state
        moe_hidden = self.moe(hidden)
        logits = self.lm_head(moe_hidden)
        return logits 