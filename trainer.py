from ppo import PPO

import torch
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', default=0, type=int, help='Graphics Device ID')
parser.add_argument('--num_envs', default=512, type=int)
parser.add_argument('--headless', default=False, type=bool)
parser.add_argument('--wandb', default=False, type=bool)
parser.add_argument('--num_steps', default=1000000, type=int)

args = parser.parse_args()

torch.manual_seed(0)
random.seed(0)

policy = PPO(args)

for _ in range(args.num_steps):
    policy.run()