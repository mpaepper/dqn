import numpy as np
import gym
from PIL import Image

import time
import random
from collections import deque

from memory import ReplayMemory, ReplayMemoryStatic

env = gym.make('MsPacmanDeterministic-v4')
state = env.reset()

def process_observation(observation):
        img = Image.fromarray(observation)
        img = img.resize((84,84)).convert('L')
        processed_observation = np.array(img)
        return processed_observation.astype('uint8').reshape([84,84,1])

stateproc = process_observation(state)

size = 10000
sample = 10000

num_plays = 32

payload = [stateproc, 0, 3, stateproc, False]

replay_memory = ReplayMemory(maxlen = size)
start_time = time.time()
for i in range(sample):
    replay_memory.store_observation(*payload)
print("Insert replay_memory: --- %s seconds ---" % (time.time() - start_time))

rm2 = ReplayMemoryStatic(size, (84, 84), num_plays)
start_time = time.time()
for i in range(sample):
    rm2.store_observation(*payload)
print("Insert rm2: --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
for i in range(sample):
    replay_memory.get_replays(num_plays)
print("Take replay_memory: --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
for i in range(sample):
    rm2.get_replays(num_plays)
print("Take rm2: --- %s seconds ---" % (time.time() - start_time))
