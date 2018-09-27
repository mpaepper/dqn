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

    def storeObservation(self, state, action, reward, nextState, gameOver):
        if (gameOver):
            reward = -1
        self.append([state, action, reward, nextState, gameOver])

    def getReplays(self, numPlays):
        replays = self.sample(numPlays)
        cols = [[],[],[],[],[]]
        for memory in replays:
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1].reshape(-1, 1), cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1,1))
