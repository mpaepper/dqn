import argparse
import numpy as np
#from gym import wrappers

from memory import ReplayMemory
from policy import EpsilonPolicy, MaxQPolicy
from agent import DQNAgent
from processor import AtariProcessor
from model import AtariDqnModel
import atari_wrappers

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='MsPacmanNoFrameskip-v4') # See environment possibilities: https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L298-L376
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--weights', type=str, default='./pacman.h5')
args = parser.parse_args()

img_size = (84, 84)
num_img_per_state = 4

env = atari_wrappers.make_atari(args.env_name) # Skip 4 frames per step, at the start of the game skip 0-30 steps
env = atari_wrappers.wrap_deepmind(env, frame_stack=True) # scale images to (84,84), keep 4 images for state, clip rewards to 1, set losing life as game_over
#env = wrappers.Monitor(env, '/tmp/pacman-randrun-1', force=True) # You can wrap it to record a video
np.random.seed(123)
env.seed(123)
num_actions = env.action_space.n

model = AtariDqnModel(num_actions=num_actions, input_shape=(img_size[0], img_size[1], num_img_per_state), learning_rate=0.00025, load_weights_file=args.weights)
memory = ReplayMemory(maxlen=1000000)
processor = AtariProcessor(input_size=img_size)

if (args.test):
    policy = MaxQPolicy()
    dqn = DQNAgent(env=env, memory=memory, policy=policy, model=model, discount_rate=0.99, processor=processor)
    dqn.play()
else:
    policy = EpsilonPolicy(epsilon_max=1.0, epsilon_min=0.1, decay_steps=1250000)
    dqn = DQNAgent(env=env, memory=memory, policy=policy, batch_size=32, model=model, discount_rate=0.99, processor=processor, weights_filename='./pacman.h5')
    dqn.fit(num_steps=4000000, start_train=1000, learn_every=4, update_target_model=5000, save_every=1000)
