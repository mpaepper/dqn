import time
import numpy as np
import tensorflow as tf

class Tracker:
    def __init__(self, window_size=50, log_dir='./tf_logs'):
        self.time = time.time()
        self.episode = 0
        self.frame = 0
        self.window_size = window_size
        self.rewards = []
        self.logs = []
        self.writer = tf.summary.FileWriter(log_dir)

    def print_episode(self, reward, frame, epsilon, keras_log_data):
        current_time = time.time()
        time_elapsed = current_time - self.time
        frame_elapsed = frame - self.frame
        speed = frame_elapsed / time_elapsed
        self.episode = self.episode + 1
        self.rewards.append(reward)
        mean_reward = np.mean(self.rewards[-self.window_size:])
        print("Frame {:d}, Episode {:d} ({:d} steps), current reward: {:5.2f}, mean reward: {:5.2f}, Frame rate: {:4.2f} f/s, Epsilon: {:.2}"
              .format(frame, self.episode, frame_elapsed, reward, mean_reward, speed, epsilon))
        episode_averages = np.mean(np.array(self.logs), axis=0)
        print("Mean reward per step: {:.2}, Mean action: {:.2}".format(episode_averages[0], episode_averages[1]))
        value_dict = {'reward': reward, 'mean_reward': mean_reward, 'speed': speed, 'epsilon': epsilon, 'step_reward': episode_averages[0], 'mean_action': episode_averages[1]}
        self.log_tensorboard(value_dict, self.episode)
        if keras_log_data:
            keras_metrics = [i for i in keras_log_data.params['metrics']]
            metric_strings = ["{}: {:.2}".format(i, keras_log_data.history[i][0]) for i in keras_metrics]
            print("Keras metrics: " + " ".join(metric_strings))
            value_dict = {i: keras_log_data.history[i][0] for i in keras_metrics}
            self.log_tensorboard(value_dict, self.episode)
        self.logs = []
        self.frame = frame
        self.time = current_time

    def log(self, reward, action):
        self.logs.append([reward, action])

    def log_tensorboard(self, value_dict, step):
        for key, value in value_dict.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
            self.writer.add_summary(summary, step)
