import numpy as np

class ReplayMemory:
    """
    A ReplayMemory saves a maximum of maxlen experiences and provides methods to store and retrieve them.
    """
    def __init__(self, maxlen=1000000):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        """
        Add data to the memory taking care of the maxlen size
        :param data:
        :return:
        """
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        """
        Retrieve batch_size number of experiences
        :param batch_size: number of experiences to retrieve
        :param with_replacement: whether to allow drawing the same experience twice (faster!)
        :return: A list of experiences
        """
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

    def store_observation(self, state, action, reward, next_state, game_over):
        """
        Store the observation of an experience.

        :param state: the state of the experience before acting
        :param action: the action taken
        :param reward: the reward received after taking the action
        :param next_state: the state of the experience after taking the action
        :param game_over: whether the game is over after taking the action
        """
        if (game_over):
            reward = -1
        self.append([state, action, reward, next_state, game_over])

    def get_replays(self, num_plays):
        """
        Get num_plays random experiences
        :param num_plays: number of experiences to get
        :return: (np.array of states, np.array of actions, np.array of rewards, np.array of next_states, np.array of game_overs)
        """
        replays = self.sample(num_plays)
        cols = [[],[],[],[],[]]
        for memory in replays:
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1,1))

class ReplayMemoryStatic:
    """
    Same as ReplayMemory to try to implement it more specific and with faster processing,
    but it seems that the ReplayMemory implementation is indeed faster.
    """
    def __init__(self, maxlen=1000000, image_size=(84,84), minibatch_size=32):
        self._memory_state = np.zeros(shape=(maxlen, image_size[0], image_size[1], 1), dtype=np.int8)
        self._memory_future_state = np.zeros(shape=(maxlen, image_size[0], image_size[1], 1), dtype=np.int8)
        self._rewards = np.zeros(shape=(maxlen, 1), dtype=np.float32)
        self._is_terminal = np.zeros(shape=(maxlen, 1), dtype=np.bool)
        self._actions = np.zeros(shape=(maxlen, 1), dtype=np.int8)

        self._mini_batch_state = np.zeros(shape=(minibatch_size, image_size[0], image_size[1], 1), dtype=np.float32)
        self._mini_batch_future_state = np.zeros(shape=(minibatch_size, image_size[0], image_size[1], 1), dtype=np.float32)

        self._mini_batch_size = minibatch_size
        self._maxlen = maxlen
        self._counter = 0

    def store_observation(self, state, action, reward, future_state, is_terminal):
        position = self._counter % self._maxlen
        self._memory_state[position,:,:,:] = state
        self._memory_future_state[position,:,:,:] = future_state
        self._rewards[position] = reward
        self._is_terminal[position] = is_terminal
        self._actions[position] = action
        self._counter += 1

    def get_replays(self, num_plays):
        ind = np.random.choice(self._maxlen, size=num_plays)

        # Avoiding a copy action as much as possible
        self._mini_batch_state[:] = self._memory_state[ind,:,:,:]
        self._mini_batch_future_state[:] = self._memory_future_state[ind,:,:,:]

        rewards = self._rewards[ind]
        is_terminal = self._is_terminal[ind]
        actions = self._actions[ind]

        return self._mini_batch_state, actions, rewards, self._mini_batch_future_state, is_terminal
