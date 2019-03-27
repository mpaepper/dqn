import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import argparse
import numpy as np
#from gym import wrappers
import gym

from memory import ReplayMemory
from policy import EpsilonPolicy, MaxQPolicy
from agent import DQNAgent
from processor import VoidProcessor
from model import FullyConnectedModel

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='CartPole-v1') # See environment possibilities: https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L298-L376
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

env = gym.make('CartPole-v1')
np.random.seed(123)
env.seed(123)
num_actions = env.action_space.n

model = FullyConnectedModel(num_actions=num_actions, neurons_per_layer=5, num_layers=2, learning_rate=0.002, load_weights_file=args.weights, use_dueling=True)

memory = ReplayMemory(maxlen=1000, game_over_bias=5)
processor = VoidProcessor()

if (args.test):
    policy = MaxQPolicy()
    dqn = DQNAgent(env=env, memory=memory, policy=policy, model=model, discount_rate=0.99, processor=processor)
    dqn.play()
else:
    policy = EpsilonPolicy(epsilon_max=1.0, epsilon_min=0.05, decay_steps=10000)
    dqn = DQNAgent(env=env, memory=memory, policy=policy, batch_size=64, model=model, discount_rate=0.99, processor=processor, weights_filename='./cartpole.h5')
    dqn.fit(num_steps=20000, start_train=1000, learn_every=1, update_target_model=100, save_every=1000, max_episode_score=500)
