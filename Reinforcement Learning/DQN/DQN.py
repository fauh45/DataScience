import random
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam

import numpy as np


class DQN:
    def __init__(self,
                 state_shape,
                 action_size,
                 update_every=100,
                 max_memory_len=2_000,
                 min_memory_size=500,
                 minibatch_size=10,
                 gamma=0.95,
                 learning_rate=0.001):
        self.state_shape = state_shape
        self.action_size = action_size

        self.memory = deque(maxlen=max_memory_len)
        self.min_memory_size = min_memory_size
        self.minibatch_size = minibatch_size

        self.gamma = gamma

        self.lr = learning_rate

        self.model = self._make_model()

        self.target_model = self._make_model()
        self.target_model.set_weights(self.model.get_weights())

        self.update_every = update_every
        self.update_count = 0

    def _make_model(self):
        init = HeUniform()

        model = Sequential()
        model.add(Input(shape=self.state_shape))
        model.add(Dense(24, activation='relu', kernel_initializer=init))
        model.add(Dense(24, activation='relu', kernel_initializer=init))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer=init))

        model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.lr), metrics=['accuracy'])
        return model

    def append_to_memory(self, transistion):
        """
        Append new transition to memory

        (state, action, reward, new state, is done)
        """
        self.memory.append(transistion)

    def get_qs(self, state):
        return self.model.predict(state.reshape([1, state.shape[0]]))[0]

    def train(self, is_terminal, step):
        if len(self.memory) < self.min_memory_size:
            return

        minibatch = random.sample(self.memory, self.minibatch_size)

        # Get the state value from random sample of action mamory
        states = np.array([transition[0] for transition in minibatch])
        # Get Q value of the state from model
        state_q_values = self.model.predict(states)

        # Get the new state from random sample of action memory
        new_states = np.array([transistion[3] for transistion in minibatch])
        # Get Q value of it
        new_values_q_state = self.target_model.predict(new_states)

        # Training data container
        X = []
        Y = []

        for index, (state, action, reward, _, is_done) in enumerate(minibatch):
            if not is_done:
                max_future_q = np.max(new_values_q_state[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs = state_q_values[index]
            current_qs[action] = new_q

            X.append(state)
            Y.append(current_qs)

        # Mini batch training
        self.model.fit(np.array(X), np.array(
            Y), batch_size=self.minibatch_size, verbose=0, shuffle=False)

        if is_terminal:
            self.update_count += 1

        # After the set amount of update time in target model
        # we set the model weights with model weights
        if self.update_count > self.update_every:
            self.target_model.set_weights(self.model.get_weights())
            self.update_count = 0
