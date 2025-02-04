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
    from src.training_loop import preprocess, _tokenizer, model, critic, optimizer_model
else:
    from .training_loop import preprocess, _tokenizer, model, critic, optimizer_model


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
    logger.info("Starting self_improve with %d candidates", num_candidates)
    model.eval()
    candidates = []
    rewards = []
    
    # Generate candidates with integrated chain-of-thought reasoning
    for _ in range(num_candidates):
        # Append chain-of-thought trigger to encourage internal reasoning
        prompt_with_cot = prompt + "\nChain-of-Thought:"
        # Increase max_length for allowing chain-of-thought reasoning
        outputs = model.generate(prompt_with_cot, max_length=100, do_sample=True)
        final_output = extract_final_answer(outputs)
        candidates.append(final_output)
    
    model.train()
    
    # Evaluate candidates with error handling and timing
    for i, cand in enumerate(candidates):
        start_eval = time.time()
        try:
            outputs = model(cand, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            reward = critic(hidden)
            rewards.append(reward)
            logger.info("Candidate %d evaluated in %.4f sec", i, time.time()-start_eval)
        except Exception as e:
            logger.error("Candidate %d evaluation failed: %s", i, e)
            rewards.append(torch.tensor(0.0))
            
    # Log candidate diversity
    try:
        reward_vals = [r.item() for r in rewards]
        diversity = max(reward_vals) - min(reward_vals)
        logger.info("Candidate rewards: %s | Diversity: %.4f", reward_vals, diversity)
    except Exception as e:
        logger.error("Error computing candidate diversity: %s", e)
    
    # Select best candidate
    try:
        best_index = torch.argmax(torch.stack(rewards))
        best_candidate = candidates[best_index]
    except Exception as e:
        logger.error("Selection of best candidate failed: %s", e)
        best_candidate = None
    
    # RL update step with error handling
    try:
        loss = -rewards[best_index].mean()
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        logger.info("RL update completed with loss: %.4f", loss.item())
    except Exception as e:
        logger.error("RL update failed: %s", e)
    
    return best_candidate


if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = preprocess(prompt_text)
    improved = self_improve(prompt)
    print(f"Improved prompt: {_tokenizer.decode(improved[0], skip_special_tokens=True)}") 