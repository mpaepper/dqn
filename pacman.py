import argparse
import numpy as np
import gym

from memory import ReplayMemory
from policy import EpsilonPolicy
from agent import DQNAgent
from processor import AtariProcessor
from model import AtariDqnModel
import atari_wrappers

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='MsPacmanNoFrameskip-v4') # See environment possibilities: https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L298-L376
args = parser.parse_args()

img_size = (84, 84)
num_img_per_state = 4

#from gym import wrappers

env = atari_wrappers.make_atari(args.env_name)
env = atari_wrappers.wrap_deepmind(env, frame_stack=True)
#env = wrappers.Monitor(env, '/tmp/pacman-randrun-1', force=True) # You can wrap it to record a video
np.random.seed(123)
env.seed(123)
num_actions = env.action_space.n

model = AtariDqnModel(num_actions=num_actions, input_shape=(img_size[0], img_size[1], num_img_per_state)).get_model()
memory = ReplayMemory(maxlen=1000000) #, window_length=num_img_per_state)
processor = AtariProcessor(input_size=img_size)
policy = EpsilonPolicy(epsilon_max=1.0, epsilon_min=0.1, decay_steps=1250000)

dqn = DQNAgent(env=env, memory=memory, policy=policy, batch_size=32, model=model, discount_rate=0.99, processor=processor)
dqn.fit(num_steps=4000000, start_train=1000, learn_every=4)
