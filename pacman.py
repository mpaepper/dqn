import argparse
import numpy as np
import gym

from memory import ReplayMemory
from policy import EpsilonPolicy
from agent import DQNAgent
from processor import AtariProcessor
from model import AtariDqnModel

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='MsPacmanNoFrameskip-v0')
args = parser.parse_args()

img_size = (84, 84)
num_img_per_state = 1


#from gym import wrappers

env = gym.make(args.env_name)
#env = wrappers.Monitor(env, '/tmp/pacman-randrun-1', force=True) # You can wrap it to record a video
np.random.seed(123)
env.seed(123)
num_actions = env.action_space.n
print(num_actions)

model = AtariDqnModel(num_actions=num_actions).get_model()
memory = ReplayMemory(maxlen=1000000) #, window_length=num_img_per_state)
processor = AtariProcessor(input_size=img_size)
policy = EpsilonPolicy(epsilon_max=1.0, epsilon_min=0.1, decay_steps=1250000)

dqn = DQNAgent(env=env, memory=memory, policy=policy, batch_size=32, model=model, discount_rate=0.99, processor=processor)
dqn.fit(num_steps=4000000, skip_start=30, start_train=50000)
