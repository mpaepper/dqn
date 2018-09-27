import time
import numpy as np

class Tracker:
    def __init__(self, window_size=50):
        self.time = time.time()
        self.episode = 0
        self.frame = 0
        self.window_size = window_size
        self.rewards = []

    def track(self, reward, frame, epsilon):
        current_time = time.time()
        time_elapsed = current_time - self.time
        frame_elapsed = frame - self.frame
        speed = frame_elapsed / time_elapsed
        self.episode = self.episode + 1
        self.rewards.append(reward)
        mean_reward = np.mean(self.rewards[-self.window_size:])
        print("Frame {:d}, Episode {:d} ({:d} steps), current reward: {:5.2f}, mean reward: {:5.2f}, Frame rate: {:4.2f} f/s, Epsilon: {:.2}"
              .format(frame, self.episode, frame_elapsed, reward, mean_reward, speed, epsilon))
        self.frame = frame
        self.time = current_time