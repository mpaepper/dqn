import numpy as np
from memory import ReplayMemory
from policy import EpsilonPolicy
from tracker import Tracker
import keras.backend as K
from keras.models import load_model, Model, Input
from keras.layers import Lambda
from keras.optimizers import Adam
import tensorflow as tf


def copy_model(model):
    model.save('./tmp_model')
    return load_model('tmp_model')


class DQNAgent:
    def __init__(self, env=None, memory=ReplayMemory(10000), policy=EpsilonPolicy(),
                 batch_size=32, model=None, discount_rate=None, processor=None, gradient_clip = 1.0):
        super(DQNAgent, self).__init__()
        self.env = env
        self.memory = memory
        self.policy = policy
        self.batch_size = batch_size
        self.dqn_model = model
        self.model = model.get_nn_model()
        self.target_model = copy_model(self.model) # The target model is used to for improved training stability -> it predicts the q values in learn()
        self.num_actions = self.dqn_model.get_num_actions()
        self.setup_trainable_model(Adam(lr=self.dqn_model.get_learning_rate), gradient_clip, self.dqn_model.get_num_actions())
        self.discount_rate = discount_rate
        self.episode_rewards = 0
        self.state = None
        self.processor = processor

    def setup_trainable_model(self, optimizer, gradient_clip, num_actions):
        model_input = self.model.input
        y_pred = self.model.output
        y_true = Input(name='y_true', batch_shape=(None, num_actions))
        masks = Input(name='masks', batch_shape=(None, num_actions))

        def avg_max_q(y_true, y_pred):
            return K.mean(K.max(y_pred, axis=-1))

        def compute_loss(inputs):
            y_pred, y_true, masks = inputs
            diff = K.abs(y_true - y_pred)
            huber_loss = tf.where(diff < 1.0, 0.5 * tf.square(diff), gradient_clip * (diff - .5 * gradient_clip)) # See https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
            masked_loss = huber_loss * masks
            return K.sum(masked_loss, axis=-1)

        loss = Lambda(compute_loss, output_shape=(1,))([y_pred, y_true, masks])
        self.trainable_model = Model(inputs=[model_input] + [y_true, masks], outputs=loss)
        self.trainable_model.compile(optimizer=optimizer, metrics=[avg_max_q], loss=[lambda y_true, y_pred: y_pred]) # loss is calculated by Lambda layer, so output y_pred
        self.trainable_model.summary()

    def act(self, current_frame):
        get_q_values = lambda: self.model.predict(self.processor.process_batch(np.array([self.state])))[0]
        action = self.policy.get_action(get_q_values, self.num_actions, current_frame)
        next_state, reward, game_over, info = self.env.step(action)
        next_state_processed = self.processor.process(next_state)
        reward_processed = self.processor.process_reward(reward)
        self.memory.store_observation(self.state, action, reward_processed, next_state_processed, game_over)
        self.state = next_state_processed
        return reward_processed, game_over, action

    def learn(self):
        states, actions, rewards, next_states, game_overs = self.memory.get_replays(self.batch_size)
        states = self.processor.process_batch(states)
        next_states = self.processor.process_batch(next_states)
        q_values_next = self.target_model.predict(np.array(next_states))
        max_q_values = np.max(q_values_next, axis=1, keepdims=True)
        expected_values = rewards + self.discount_rate * max_q_values * ~game_overs
        y_true = np.zeros((self.batch_size, self.num_actions))
        masks = np.zeros((self.batch_size, self.num_actions))
        for idx, action in enumerate(actions):
            masks[idx, action] = 1.
            y_true[idx, action] = expected_values[idx]
        return self.trainable_model.fit([states, y_true, masks], y_true, verbose=0)

    def fit(self, num_steps=4000000, start_train=50000, max_episode_score=1000, learn_every=4, update_target_model=10000):
        tracker = Tracker()
        self.start_new_episode()
        game_over = False
        keras_log_data = []
        for i in range(num_steps):
            if game_over or self.episode_rewards >= max_episode_score:
                tracker.print_episode(self.episode_rewards, i, self.policy.get_epsilon(i), keras_log_data)
                self.start_new_episode()
            reward, game_over, action = self.act(i)
            tracker.log(reward, action)
            self.episode_rewards += reward

            if i >= start_train:
                if i % learn_every == 0:
                    keras_log_data = self.learn()
                if i % update_target_model == 0:
                    self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def start_new_episode(self):
        self.state = self.processor.process(self.env.reset())
        # Skipping of start steps is done by NoopResetEnv wrapper in environment
        self.episode_rewards = 0
