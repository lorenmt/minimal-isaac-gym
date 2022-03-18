import torch
import collections
import random

"""
    A simple Random Replay Buffer.
"""


class ReplayBuffer:
    def __init__(self, buffer_limit=int(1e6), num_envs=1):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.num_envs = num_envs

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append(tuple([obs, action, reward, next_obs, done]))

    def sample(self, mini_batch_size):
        obs, action, reward, next_obs, done = zip(*random.sample(self.buffer, mini_batch_size))

        rand_idx = torch.randperm(mini_batch_size * self.num_envs)  # random shuffle tensors

        obs = torch.cat(obs)[rand_idx]
        action = torch.cat(action)[rand_idx]
        reward = torch.cat(reward)[rand_idx]
        next_obs = torch.cat(next_obs)[rand_idx]
        done = torch.cat(done)[rand_idx]
        return obs, action, reward, next_obs, done

    def size(self):
        return len(self.buffer)
