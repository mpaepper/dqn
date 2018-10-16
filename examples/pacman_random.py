import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import argparse
import numpy as np

from policy import RandomPolicy
from agent import DQNAgent
from processor import AtariProcessor
from model import AtariDqnModel
import atari_wrappers

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='MsPacmanNoFrameskip-v4') # See environment possibilities: https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L298-L376
args = parser.parse_args()

img_size = (84, 84)
num_img_per_state = 4

env = atari_wrappers.make_atari(args.env_name) # Skip 4 frames per step, at the start of the game skip 0-30 steps
env = atari_wrappers.wrap_deepmind(env, frame_stack=True) # scale images to (84,84), keep 4 images for state, clip rewards to 1, set losing life as game_over
num_actions = env.action_space.n

model = AtariDqnModel(num_actions=num_actions, input_shape=(img_size[0], img_size[1], num_img_per_state))
dqn = DQNAgent(env=env, policy=RandomPolicy(), model=model, discount_rate=0.99, processor=AtariProcessor())
dqn.play()
