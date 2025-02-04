from torch.optim import Adam
from src.transformer_model import model
from src.critic import Critic
import torch  # added for self_improve function
import torch.nn.functional as F  # added for critic update
import logging
from transformers import GPT2Tokenizer

logging.basicConfig(level=logging.INFO)
_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

critic = Critic()
optimizer_model = Adam(model.parameters(), lr=1e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-4)  # added critic optimizer

# Added function to update critic parameters based on evaluation (Task 36)
def update_critic(cand):
    bh = model(cand, output_hidden_states=True).hidden_states[-1]
    target = torch.ones(1, device=bh.device)  # baseline target
    loss = F.mse_loss(critic(bh).mean(), target)
    optimizer_critic.zero_grad()
    loss.backward()
    optimizer_critic.step()

# Task 38-39: Integrated candidate evaluation within a loop
def self_improve(prompt, num_candidates=5):
    model.eval()
    candidates, rewards = [], []
    for _ in range(num_candidates):
        try:
            with torch.no_grad():
                cand = model.generate(prompt, max_length=50, do_sample=True)
            if cand is None:
                raise ValueError("Generation returned None")
        except Exception as e:
            logging.error("Generation failed: %s", e)
            continue
        out = model(cand, output_hidden_states=True)
        candidates.append(cand)
        rewards.append(critic(out.hidden_states[-1]).mean())
    if not candidates:
        raise RuntimeError("No valid candidates generated")
    for i, cand in enumerate(candidates):
        logging.info("Candidate %d: %s, reward: %.4f", i, _tokenizer.decode(cand[0], skip_special_tokens=True), rewards[i].item())
    best_index = torch.argmax(torch.stack(rewards)).item()
    logging.info("Selected candidate %d with reward: %.4f", best_index, rewards[best_index].item())
    best_candidate = candidates[best_index]

    model.train()
    outputs = model(best_candidate, labels=best_candidate)
    loss_rl = rewards[best_index] * outputs.loss
    optimizer_model.zero_grad()
    loss_rl.backward()
    optimizer_model.step()

    update_critic(best_candidate)
    return best_candidate

if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = _tokenizer.encode(prompt_text, return_tensors='pt')
    improved = self_improve(prompt)
    print(f"Improved prompt: {_tokenizer.decode(improved[0], skip_special_tokens=True)}") 