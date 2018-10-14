from PIL import Image
import numpy as np


class AtariProcessor:
    """
    The AtariProcessor can manipulate states and rewards.
    """

    def process(self, state):
        """Process the given state"""
        return state # Conversion to input_size is already handled by the environment (WarpFrame wrapper)

    def process_batch(self, batch):
        """Process the given batch to convert from int to float values"""
        processed_batch = batch.astype('float32') / 255. # Convert int to float32, b/c int is 4 times smaller in storage
        return processed_batch

    def process_reward(self, reward):
        """Process the given reward"""
        return reward # Reward clipping is handled by the environment (ClipRewardEnv wrapper)

class VoidProcessor:
    """
    The VoidProcessor does nothing
    """

    def process(self, state):
        return state

    def process_batch(self, batch):
        return batch

    def process_reward(self, reward):
        return reward
