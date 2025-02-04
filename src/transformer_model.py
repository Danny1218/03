from transformer_criteria import get_config
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel(get_config())
print(model) 