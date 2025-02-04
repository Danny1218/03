# Self-Improvement & Evaluation Process

Our model generates multiple candidate responses from a given prompt using sampling. Each candidate is scored by a critic network that averages its final hidden states to yield a reward.

The highest-reward candidate is refined via a minimal Monte Carlo Tree Search expansion, then used in a reinforcement learning update to adjust both the transformer and critic.

This streamlined loop ensures continuous model improvement and effective critic calibration. 