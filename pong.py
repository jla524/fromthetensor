"""
Train a model to play pong
"""
import gym

env = gym.make('Pong-v4')
observation = env.reset()
