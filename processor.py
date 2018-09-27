from PIL import Image
import numpy as np

class AtariProcessor:
    def __init__(self, input_size=(84, 84)):
       self.input_size = input_size

    def process(self, state):
        img = Image.fromarray(state)
        img = img.resize(self.input_size).convert('L')
        processed_observation = np.array(img)
        return processed_observation.astype('uint8').reshape(self.input_size[0], self.input_size[1], 1) # TODO Reshape only as workaround

    def process_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)