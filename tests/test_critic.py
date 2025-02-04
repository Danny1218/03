import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.critic import Critic


def test_critic_shape():
    critic = Critic()
    sample = torch.randn(3, 7, 128)  # batch=3, sequence length=7
    out = critic(sample)
    assert out.shape == (3, 1)


def test_critic_output_valid():
    critic = Critic()
    sample = torch.ones(2, 5, 128)
    out = critic(sample)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all() 