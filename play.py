from isaacgym import gymapi
from isaacgym import gymtorch
from bolt_env import BoltEnv
from ppo import Net
import torch
import numpy as np
import torch.nn as nn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', default=0, type=int, help='Graphics Device ID')
parser.add_argument('--num_envs', default=1, type=int)
parser.add_argument('--headless', default=False, type=bool)
parser.add_argument('--wandb', default=False, type=bool)

args = parser.parse_args()

env = BoltEnv(args)
net = Net(env.num_obs, env.num_act).to(args.sim_device)
weight_dict = torch.load('best.pth')
net.load_state_dict(weight_dict)

while True:
    obs = env.obs_buf.clone()

    with torch.no_grad():
        mu = net.pi(obs)
        action = mu.clip(-1, 1)
    
    env.step(action)