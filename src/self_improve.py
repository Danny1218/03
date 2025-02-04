from torch.optim import Adam
from transformer_model import model
from critic import Critic
import torch  # added for self_improve function

critic = Critic()
optimizer_model = Adam(model.parameters(), lr=1e-4)

# Task 26: Self-consistency via candidate sampling and Task 31: Selecting best candidate with highest reward
def self_improve(prompt, num_candidates=5):
    model.eval()
    with torch.no_grad():
        cands = [model.generate(prompt, max_length=50, do_sample=True) for _ in range(num_candidates)]
    model.train()
    # Compute rewards using critic on extracted hidden states
    rewards = torch.stack([critic(model(c, output_hidden_states=True).hidden_states[-1]).mean() for c in cands])
    best_idx = torch.argmax(rewards)  # Task 31: select index with the highest reward
    best = cands[best_idx]
    loss = -rewards[best_idx]  # Task 33: perform backpropagation on the selected candidate's loss
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