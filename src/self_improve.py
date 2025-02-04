from torch.optim import Adam
from transformer_model import model
from critic import Critic
import torch  # added for self_improve function

critic = Critic()
optimizer_model = Adam(model.parameters(), lr=1e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-4)

# Task 26: Self-consistency via candidate sampling
def generate_candidates(prompt, num_candidates=5):
    model.eval()
    with torch.no_grad():
        candidates = [model.generate(prompt, max_length=50, do_sample=True) for _ in range(num_candidates)]
    model.train()
    return candidates

# Task 28: Extract hidden states from candidate outputs
def extract_hidden(candidate):
    return model(candidate, output_hidden_states=True).hidden_states[-1]

def compute_rewards(candidates):
    # Task 29 & 30: Compute rewards for each candidate response using the critic
    return torch.stack([critic(extract_hidden(c)).mean() for c in candidates])

# Task 26 & 28 & 29: Incorporated self-consistency, hidden state extraction and candidate evaluation
def self_improve(prompt, num_candidates=5):
    cands = generate_candidates(prompt, num_candidates)
    rewards = compute_rewards(cands)
    best = cands[torch.argmax(rewards)]
    loss = -torch.max(rewards)
    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()
    return best

if __name__ == '__main__':
    import sys
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = tokenizer.encode(prompt_text, return_tensors='pt')
    improved = self_improve(prompt)
    print(tokenizer.decode(improved[0], skip_special_tokens=True)) 