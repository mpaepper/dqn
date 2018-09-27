import numpy as np
from memory import ReplayMemory
from policy import EpsilonPolicy
from tracker import Tracker

class DQNAgent:
    def __init__(self, env=None, memory=ReplayMemory(10000), policy=EpsilonPolicy(),
                 batch_size=32, model=None, discount_rate=None, processor=None):
        super(DQNAgent, self).__init__()
        self.env = env
        self.memory = memory
        self.policy = policy
        self.batch_size = batch_size
        self.model = model
        self.discount_rate = discount_rate
        self.episode_rewards = 0
        self.state = None
        self.processor = processor

    def act(self, current_frame):
        q_values = self.model.predict(self.processor.process_batch(np.array([self.state])))
        action = self.policy.get_action(q_values, current_frame) # TODO: q_values are only needed if epsilon policy does NOT act randomly!
        next_state, reward, game_over, info = self.env.step(action)
        next_state_processed = self.processor.process(next_state)
        reward_processed = self.processor.process_reward(reward)
        self.memory.store_observation(self.state, action, reward_processed, next_state_processed, game_over)
        self.state = next_state_processed
        return reward_processed, game_over

    def learn(self):
        states, actions, rewards, next_states, game_overs = self.memory.get_replays(self.batch_size)
        states = self.processor.process_batch(states)
        next_states = self.processor.process_batch(next_states)
        q_values_next = self.model.predict(np.array(next_states))
        max_q_values = np.max(q_values_next, axis=1, keepdims=True)
        expectedValues = rewards + self.discount_rate * max_q_values
        actual_values = self.model.predict(np.array(states))
        for idx, i in enumerate(actions):
            if (game_overs[idx]):
                actual_values[idx, i] = rewards[idx]
            else:
                actual_values[idx, i] = expectedValues[idx]
        self.model.fit(states, actual_values, verbose=0)

    def fit(self, num_steps=4000000, skip_start=30, start_train=50000, max_episode_score=1000, learn_every=4):
        tracker = Tracker()
        self.start_new_episode(skip_start)
        game_over = False
        for i in range(num_steps):
            if game_over or self.episode_rewards >= max_episode_score:
                tracker.track(self.episode_rewards, i, self.policy.get_epsilon(i))
                self.start_new_episode(skip_start)
            reward, game_over = self.act(i)
            self.episode_rewards += reward

            if i >= start_train and i % learn_every == 0:
                self.learn()

    def start_new_episode(self, skip_start):
        self.env.reset()
        for skip in range(skip_start):
            state, reward, _, _ = self.env.step(0)
        self.state = self.processor.process(state)
        self.episode_rewards = 0
