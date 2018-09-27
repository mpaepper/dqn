import numpy as np

class EpsilonPolicy:
    def __init__(self, epsilon_max = 1.0, epsilon_min = 0.1, decay_steps = 1000000):
        super(EpsilonPolicy, self).__init__()
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_steps = decay_steps

    def get_action(self, q_values, frame):
        if np.random.random() <= self.get_epsilon(frame):
            num_actions = len(q_values)
            return np.random.choice(num_actions)
        else:
            return np.argmax(q_values)

    def get_epsilon(self, frame):
        return max(self.epsilon_min, self.epsilon_max - (self.epsilon_max - self.epsilon_min) * frame / self.decay_steps)