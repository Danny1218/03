from torch.optim import Adam
from src.transformer_model import model
from src.critic import Critic
import torch  # added for self_improve function
import torch.nn.functional as F  # added for critic update
import logging
from transformers import GPT2Tokenizer

logging.basicConfig(level=logging.INFO)
_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def preprocess(text):
    return _tokenizer.encode(text, return_tensors='pt')

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

# Added minimal MCTS expansion function
def mcts_expand(node, simulations=2):
    children, rewards = [], []
    for _ in range(simulations):
        try:
            with torch.no_grad():
                child = model.generate(node, max_length=50, do_sample=True)
            out = model(child, output_hidden_states=True)
            children.append(child)
            rewards.append(critic(out.hidden_states[-1]).mean())
        except Exception as e:
            logging.error("MCTS error: %s", e)
    if children:
        best_idx = torch.argmax(torch.stack(rewards)).item()
        return children[best_idx], rewards[best_idx]
    return node, None

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

    # MCTS expansion step
    best_candidate, mcts_r = mcts_expand(best_candidate, simulations=2)
    if mcts_r is not None:
        logging.info("MCTS updated candidate reward: %.4f", mcts_r.item())

    final_reward = mcts_r if mcts_r is not None else rewards[best_index]
    model.train()
    outputs = model(best_candidate, labels=best_candidate)
    loss_rl = final_reward * outputs.loss
    optimizer_model.zero_grad()
    loss_rl.backward()
    optimizer_model.step()

    update_critic(best_candidate)
    return best_candidate

if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = preprocess(prompt_text)  # Use the preprocess function
    improved = self_improve(prompt)
    print(f"Improved prompt: {_tokenizer.decode(improved[0], skip_special_tokens=True)}") 