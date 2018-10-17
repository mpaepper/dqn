# Keras / Tensorflow library for deep Q learning

This library can be used to train an agent using deep Q learning.
It follows the default CNN architecture used in the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602).

A reinforcement agent needs an environment/game in which to play, a memory (where and how to store experiences), a policy (how to act), a model (the neural network to be trained) and a processor (this is for mainly to save some memory in the storage) which you need to pass the agent on creation.

There are some examples of how to pass these items to the agent in the examples/ folder:

## Cartpole

An easy environment to try out is the cartpole environment which has only 4 state variables and is thus fast to train and easy to play with.

* To execute a random episode of cartpole where the agent is acting randomly, you can run "python examples/cartpole_random.py".
* To train an agent to play cartpole for 10.000 steps, simply run "python examples/cartpole.py". This should be very fast to run (around a minute). It will output some debug information of the progress.
* After training, you can test your cartpole agent by running: "python examples/cartpole.py --weights=cartpole.h5 --test=True".

The cartpole architecture uses two layers of five neurons and a final layer of two neurons for the two possible actions.

## Pacman

* To see Pacman doing random actions, you can run "python examples/pacman_random.py".
* To train an agent playing the Atari game of MsPacman, simply run "python examples/pacman.py" which will run for 4 million steps, so beware that this will take quite a while (around 12 hours using a GPU).
* To test your learned agent afterwards, run: "python examples/pacman.py --weights=pacman.h5 --test=True". This will run an episode and visualize that episode.
