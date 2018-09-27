from PIL import Image
import numpy as np

class AtariProcessor:
    def __init__(self, input_size=(84, 84)):
       self.input_size = input_size

    def process(self, state):
        return state # Conversion to input_size is already handled by the environment (WarpFrame wrapper)

    def process_batch(self, batch):
        processed_batch = batch.astype('float32') / 255. # Convert int to float32, b/c int is 4 times smaller in storage
        return processed_batch

    def process_reward(self, reward):
        return reward # Reward clipping is handled by the environment (ClipRewardEnv wrapper)