import gym
import tensorflow as tf
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras import losses
from keras.layers import LeakyReLU

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.policy import LinearAnnealedPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'LunarLander-v2'

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(152)
env.seed(251)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('elu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
print(model.summary())

#policy = EpsGreedyQPolicy()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr = 'eps', value_max = 1, value_min = .01, nb_steps = 10000, value_test= .001)
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=500,
target_model_update=1e-1, policy=policy)
dqn.compile(optimizer = Adam(lr=1e-4), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

dqn.test(env, nb_episodes=10, visualize=True)

dqn.save_weights('/home/alex/model_lunar_lander.h5')

