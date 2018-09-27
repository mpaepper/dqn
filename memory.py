import numpy as np

class ReplayMemory:
    def __init__(self, maxlen=1000000):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

    def store_observation(self, state, action, reward, next_state, game_over):
        if (game_over):
            reward = -1
        self.append([state, action, reward, next_state, game_over])

    def get_replays(self, numPlays):
        replays = self.sample(numPlays)
        cols = [[],[],[],[],[]]
        for memory in replays:
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1].reshape(-1, 1), cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1,1))

class ReplayMemoryStatic:
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
