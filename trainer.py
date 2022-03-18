# from ppo import PPO
from policy import Policy

import torch
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
parser.add_argument('--num_envs', default=512, type=int)
parser.add_argument('--headless', action='store_true')

args = parser.parse_args()

torch.manual_seed(0)
random.seed(0)

# env = Cartpole(args)
policy = Policy(args)

# ppo = PPO(args)


while True:
    policy.run()

# a = []
# while True:
#     act = 2 * (torch.rand(args.num_envs, device=args.sim_device) - 0.5)
#     env.step(act)
#     # print(env.obs_buf, env.reward_buf, env.reset_buf)
#     a.append(env.obs_buf)
