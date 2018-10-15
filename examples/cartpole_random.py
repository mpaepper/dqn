import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import argparse
import numpy as np
#from gym import wrappers
import gym

from memory import ReplayMemory
from policy import EpsilonPolicy, RandomPolicy
from agent import DQNAgent
from processor import VoidProcessor
from model import FullyConnectedModel

parser = argparse.ArgumentParser()
args = parser.parse_args()

env = gym.make('CartPole-v1')
num_actions = env.action_space.n

model = FullyConnectedModel(num_actions=num_actions, neurons_per_layer=5, num_layers=2, learning_rate=0.002, load_weights_file=None)

memory = ReplayMemory(maxlen=10000, game_over_bias=10)
processor = VoidProcessor()

policy = RandomPolicy()
dqn = DQNAgent(env=env, memory=memory, policy=policy, model=model, discount_rate=0.99, processor=processor)
dqn.play(delay=0.2)
