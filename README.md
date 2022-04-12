# Minimal Isaac Gym
This repository provides a minimal example of NVIDIA's [Isaac Gym](https://developer.nvidia.com/isaac-gym), to assist other researchers like me to quickly understand the code structure, to be able to design fully customised large-scale reinforcement learning experiments.

The example is based on the official implementation from the Isaac Gym's [Benchmark Experiments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), for which we have followed a similar implementation on [Cartpole](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/cartpole.py), but with a minimal number of lines of code aiming for maximal readability, and without using any third-party RL frameworks. 

**Note**: The current implementation is based on Isaac Gym Preview Version 3, with the support for two RL algorithms: *DQN* and *PPO* (both continuous and discrete version). PPO seems to be the default RL algorithm for Isaac Gym from the recent works of [Learning to walk](https://arxiv.org/abs/2109.11978) and [Object Re-orientation](https://arxiv.org/abs/2111.03043), since it only requires on-policy training data and therefore to make it a much simpler implementation coupled with Isaac Gym's APIs. 

*Both DQN and PPO are expected to converge under 1 minute.*

## Usage
Simply run `python trainer.py --method {dqn; ppo, ppo_d}`.

## Disclaimer
I am also very new to Isaac Gym, and I cannot guarantee my implementation is absolutely correct. If you have found anything unusual or unclear that can be improved, PR or Issues are highly welcomed.

