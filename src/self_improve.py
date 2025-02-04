import logging
import time
import torch

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

if __package__ is None:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from src.training_loop import preprocess, _tokenizer, model, critic, optimizer_model, optimizer_critic
else:
    from .training_loop import preprocess, _tokenizer, model, critic, optimizer_model, optimizer_critic


def retrieve_knowledge(prompt):
    # Minimal RAG: using hardcoded external knowledge
    return "External knowledge: Latest research shows retrieval augments reasoning and factual correctness."


def retrieve_knowledge_context(prompt):
    # Minimal stub for external retrieval.
    # In production, replace with API call or knowledge base query.
    return "External context: Recent AI advances show improved factual grounding with retrieval augmentation."


# Added helper function for extracting final answer from chain-of-thought generation
def extract_final_answer(generated_text: str) -> str:
    # If 'Final Answer:' marker exists, extract text after it; otherwise, return full text
    if 'Final Answer:' in generated_text:
        return generated_text.split('Final Answer:')[-1].strip()
    return generated_text.strip()


def self_improve(prompt, num_candidates=5):
    retrieved = retrieve_knowledge(prompt)  # Retrieve external knowledge
    augmented_prompt = retrieved + "\n" + str(prompt)  # Augment prompt with external info
    model.eval()
    candidates = []
    rewards = []
    
    # Generate multiple candidate responses using chain-of-thought reasoning:
    chain_prompt = augmented_prompt + "\nPlease provide your chain-of-thought reasoning, step-by-step, and then output your Final Answer in the format 'Final Answer: <answer>'"
    for _ in range(num_candidates):
        outputs = model.generate(chain_prompt, max_length=150, do_sample=True)
        final_text = extract_final_answer(outputs)
        candidates.append(final_text)
    
    model.train()
    
    # Compute rewards for each candidate using the critic
    for cand in candidates:
        input_ids = _tokenizer.encode(cand, return_tensors='pt')
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        candidate_reward = critic(hidden.mean(dim=1))
        rewards.append(candidate_reward)

    # Voting system: select candidate with highest pairwise word overlap
    def vote_score(text):
        words = set(text.split())
        return sum(len(words & set(other.split())) for other in candidates if other != text)

    best_index = torch.argmax(torch.stack(rewards))
    best_candidate = candidates[best_index]

    # Direct Preference Optimization RL update: pairwise ranking loss
    stacked_rewards = torch.stack(rewards)  # assumed shape: (num_candidates, 1)
    loss = 0
    num_pairs = 0
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            diff = stacked_rewards[i] - stacked_rewards[j]
            loss += -torch.log(torch.sigmoid(diff))
            num_pairs += 1
    loss /= num_pairs
    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()

    # Critic update using simulated human feedback: best candidate is preferred (target=1), others 0
    target = torch.zeros(len(candidates), 1)
    target[best_index] = 1.0
    critic_loss = torch.nn.BCEWithLogitsLoss()(stacked_rewards, target)
    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    return best_candidate


if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = preprocess(prompt_text)
    improved = self_improve(prompt)
    print(f"Improved prompt: {improved}") 