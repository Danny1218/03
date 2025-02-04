# Define criteria for a small transformer model
CRITERIA = {"max_params":50e6, "embedding_size":128, "num_layers":4, "num_heads":4, "vocab_size":50257}

# Task 12 complete: embedding size set to 128 and layer count set to 4

def get_config():
    from transformers import GPT2Config
    return GPT2Config(n_embd=CRITERIA["embedding_size"], n_layer=CRITERIA["num_layers"], n_head=CRITERIA["num_heads"], vocab_size=CRITERIA["vocab_size"]) 