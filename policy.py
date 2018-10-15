import numpy as np

class EpsilonPolicy:
    """
    EpsilonPolicy to choose actions. It chooses a random action with probability epsilon
    and the best action as estimated by the Q values with (1-epsilon)
    Arguments:
        epsilon_max -- Initial value of epsilon
        epsilon_min -- The minimum, final value of epsilon
        decay_steps -- Number of steps during which epsilon is decayed from epsilon_max to epsilon_min
    """
    def __init__(self, epsilon_max = 1.0, epsilon_min = 0.1, decay_steps = 1000000):
        super(EpsilonPolicy, self).__init__()
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_steps = decay_steps

    def get_action(self, get_q_values_func, num_actions, frame):
        """
        Get the action using the epsilon policy
        :param get_q_values_func: a function which is able to retrieve the q_values of the current experience
        :param num_actions: the number of possible actions to take
        :param frame: the current step number to calculate epsilon for the decay
        :return: the action to take (in range(num_actions))
        """
        if np.random.random() <= self.get_epsilon(frame):
            return np.random.choice(num_actions)
        else:
            q_values = get_q_values_func()
            return np.argmax(q_values)

    def get_epsilon(self, frame):
        """Get the current epsilon value as calculated by the current frame step"""
        return max(self.epsilon_min, self.epsilon_max - (self.epsilon_max - self.epsilon_min) * frame / self.decay_steps)

class MaxQPolicy:
    """
    MaxQPolicy chooses the best action given the Q values.
    """
    def get_action(self, get_q_values_func, num_actions, frame):
        """
        Get the action using the maximum Q values.
        :param get_q_values_func: a function which is able to retrieve the q_values of the current experience
        :param num_actions: not used by this policy
        :param frame: not used by this policy
        :return: the action to take (in range(num_actions))
        """
        q_values = get_q_values_func()
        return np.argmax(q_values)

class RandomPolicy:
    """
    RandomPolicy chooses a random action each time
    """
    def get_action(self, get_q_values_func, num_actions, frame):
        return np.random.choice(num_actions)
