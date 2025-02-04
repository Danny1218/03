import torch
import random
import numpy as np
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

if __package__ is None:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import logging
import logging.config
import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import GPT2Tokenizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.transformer_model import model
from src.critic import Critic
from src.config import LEARNING_RATE, NUM_CANDIDATES, MAX_NEW_TOKENS, MCTS_SIMS, MODEL_NAME, LOG_LEVEL, LOG_FILE

# Setup logging and tokenizer
_tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
_tokenizer.pad_token = _tokenizer.eos_token
logging_config = {
    'version': 1,
    'formatters': {
         'default': {
              'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
         }
    },
    'handlers': {
         'console': {
              'class': 'logging.StreamHandler',
              'formatter': 'default',
              'level': LOG_LEVEL
         },
         'file': {
              'class': 'logging.FileHandler',
              'formatter': 'default',
              'filename': LOG_FILE,
              'level': LOG_LEVEL
         }
    },
    'root': {
         'handlers': ['console', 'file'],
         'level': LOG_LEVEL
    }
}
logging.config.dictConfig(logging_config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Move model to device
model.to(device)

# Initialize critic and move to device
critic = Critic()
critic.to(device)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/tensorboard_logs")
global_step = 0

# Initialize the tokenizer globally
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def preprocess(text):
    tokens = _tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    return {k: v.to(device) for k, v in tokens.items()}

# Initialize optimizers
optimizer_model = Adam(model.parameters(), lr=LEARNING_RATE)
optimizer_critic = Adam(critic.parameters(), lr=LEARNING_RATE)

# Added scheduler initialization for stability improvements
scheduler_model = StepLR(optimizer_model, step_size=10, gamma=0.95)
scheduler_critic = StepLR(optimizer_critic, step_size=10, gamma=0.95)

# Initialize GradScaler
scaler = GradScaler()

def update_critic(cand, target_reward):
    bh = model(cand, output_hidden_states=True).hidden_states[-1]
    optimizer_critic.zero_grad()
    loss = F.mse_loss(critic(bh).mean(), target_reward.detach().mean())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    optimizer_critic.step()
    scheduler_critic.step()

# Modified save_checkpoint to include metrics
def save_checkpoint(step=None, metrics=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_model_state": optimizer_model.state_dict(),
        "optimizer_critic_state": optimizer_critic.state_dict(),
        "metrics": metrics,
    }
    fname = f"checkpoint_{step}.pt" if step is not None else "checkpoint.pt"
    torch.save(checkpoint, fname)


def mcts_expand(node, sims=MCTS_SIMS):
    children = [model.generate(node, attention_mask=torch.ones_like(node), max_new_tokens=MAX_NEW_TOKENS, do_sample=True, top_k=50, top_p=0.95, pad_token_id=_tokenizer.eos_token_id) for _ in range(sims)]
    rewards = [critic(model(child, output_hidden_states=True).hidden_states[-1]).mean() for child in children]
    best_index = torch.argmax(torch.stack(rewards))
    return children[best_index], rewards[best_index]

# New helper functions for modularization

def generate_candidates(model, prompt, num_candidates=5, max_length=50):
    # Generate candidate responses using sampling
    model.eval()
    candidates = []
    for _ in range(num_candidates):
        outputs = model.generate(prompt['input_ids'], attention_mask=prompt['attention_mask'], max_length=max_length, do_sample=True, pad_token_id=_tokenizer.eos_token_id)
        candidates.append(outputs)
    return candidates


def evaluate_candidates(model, critic, candidates):
    # Evaluate generated candidates and compute a reward for each
    rewards = []
    model.train()  # switch back for gradient computation
    for cand in candidates:
        outputs = model(cand, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # last layer hidden states
        reward = critic(hidden)
        rewards.append(reward)
    return rewards


def update_model(optimizer, reward):
    # Perform a simple policy gradient update using the negative reward
    loss = -reward.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


# Updated self_improve function using modularized components

def self_improve(prompt, num_candidates=5):
    global global_step
    # If prompt is a string, tokenize it; if already tokenized, use it directly.
    if isinstance(prompt, dict) and 'input_ids' in prompt and 'attention_mask' in prompt:
         input_ids = prompt['input_ids']
         attn_mask = prompt['attention_mask']
    else:
         input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
         attn_mask = torch.ones_like(input_ids)
    model.eval()
    candidates = []

    # Best-of-N sampling candidates
    sampling_candidates = generate_candidates(model, {"input_ids": input_ids, "attention_mask": attn_mask}, num_candidates=num_candidates)
    candidates.extend(sampling_candidates)

    # Beam search candidate
    beam_candidate = model.generate(input_ids,
                                    attention_mask=attn_mask,
                                    num_beams=5,
                                    early_stopping=True,
                                    max_length=MAX_NEW_TOKENS,
                                    pad_token_id=_tokenizer.eos_token_id)
    candidates.append(beam_candidate)

    # Lookahead candidate via MCTS expansion
    lookahead_candidate, _ = mcts_expand(input_ids)
    candidates.append(lookahead_candidate)

    # Evaluate candidates
    rewards = evaluate_candidates(model, critic, candidates)
    combined_rewards = torch.stack([r.mean() for r in rewards])
    best_index = torch.argmax(combined_rewards)
    best_candidate = candidates[best_index]

    # Sequential revision loop: if candidate reward is below a threshold, iteratively refine using mcts_expand
    REVISION_REWARD_THRESHOLD = 0.1  # desired minimum reward
    MAX_REVISION_STEPS = 3
    revised_reward = rewards[best_index]
    revision_step = 0
    candidate = best_candidate
    while revised_reward < REVISION_REWARD_THRESHOLD and revision_step < MAX_REVISION_STEPS:
        candidate, revised_reward = mcts_expand(candidate)
        logging.info("Revision step %d: candidate reward improved to %.4f", revision_step+1, revised_reward.item())
        revision_step += 1
    best_candidate = candidate
    
    baseline = torch.stack(rewards).mean()  # Compute baseline reward
    advantage = rewards[best_index] - baseline  # Advantage
    outputs = model(best_candidate, labels=best_candidate)  # Compute log probability loss
    loss = advantage * outputs.loss  # REINFORCE with baseline loss
    optimizer_model.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer_model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer_model)
    scaler.update()
    scheduler_model.step()
    update_critic(best_candidate, rewards[best_index])

    # Prepare metrics dictionary for checkpointing
    metrics = {
        "candidate_rewards": {
            "avg": sum(rewards)/len(rewards),
            "min": min(rewards),
            "max": max(rewards),
        },
        "candidate_losses": {
            "avg": loss.item(),
            "min": loss.item(),
            "max": loss.item(),
        }
    }

    save_checkpoint(metrics=metrics)

    # Logging metrics
    writer.add_scalar('Loss', loss.item(), global_step)
    writer.add_scalar('BestReward', rewards[best_index].item(), global_step)
    var_diversity = len(set([str(c) for c in candidates]))
    writer.add_scalar('CandidateDiversity', var_diversity, global_step)
    global_step += 1

    # Added TensorBoard logging
    writer.add_scalar("CandidateRewards/Average", sum(rewards)/len(rewards), global_step)
    writer.add_scalar("Metrics/Perplexity", torch.exp(outputs.loss).item(), global_step)

    # Logging metrics to TensorBoard
    writer.add_scalar("Loss/RL", loss.item(), global_step)
    writer.add_scalar("Loss/Total", loss.item(), global_step)
    writer.add_scalar("Metrics/Perplexity", torch.exp(outputs.loss).item(), global_step)
    writer.add_scalar("Reward/Average", sum(rewards)/len(rewards), global_step)
    global_step += 1

    return tokenizer.decode(best_candidate[0], skip_special_tokens=True)

if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = preprocess(prompt_text)
    improved = self_improve(prompt)
    print(f"Improved prompt: {improved}")

# At the end of the training, close the TensorBoard writer
writer.close()

# After training, apply dynamic quantization for faster inference
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)  # quantize linear layers 