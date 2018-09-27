from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation
from keras.optimizers import Adam


class AtariDqnModel:
    def __init__(self, input_shape=(84, 84, 1), data_format='channels_last', num_actions=9, learning_rate=0.00025, show_summary=True):
        player = Sequential()
        player.add(
            Conv2D(32, input_shape=input_shape, kernel_size=(8, 8), strides=(4, 4), data_format=data_format))
        player.add(Activation('relu'))
        player.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), data_format=data_format))
        player.add(Activation('relu'))
        player.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), data_format=data_format))
        player.add(Activation('relu'))
        player.add(Flatten())
        player.add(Dense(512))
        player.add(Activation('relu'))
        player.add(Dense(num_actions, activation='linear'))
        player.compile(optimizer=Adam(lr=learning_rate), loss='mse')
        if show_summary:
            player.summary()
        self.model = player

    def get_model(self):
        return self.model