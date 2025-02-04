import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from transformers import GPT2Tokenizer
from src.self_improve import self_improve


def test_self_improve():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt = tokenizer.encode('Synthetic prompt for testing', return_tensors='pt')
    output = self_improve(prompt)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    assert decoded, 'Output should be a non-empty string'


if __name__ == '__main__':
    test_self_improve()
    print('Test passed.') 