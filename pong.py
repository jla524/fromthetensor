"""
Train a model to play pong
"""
import os

import gym
import numpy as np
from torch import nn

# Hyperparameters
RENDER = os.getenv('RENDER') is not None
RESUME = os.getenv('RESUME') is not None

# Model initialization
NUM_HIDDEN = 200
DIMENSIONS = 80 * 80


class PongNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(NUM_HIDDEN, DIMENSIONS)
        self.layer2 = nn.Linear(DIMENSIONS)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, data):
        data = self.layer1(data)
        data = self.act1(data)
        data = self.layer2(data)
        data = self.act2(data)
        return data


# Game related
if RENDER:
    env = gym.make('ALE/Pong-v5', render_mode='human')
else:
    env = gym.make('ALE/Pong-v5')
observation = env.reset()
previous = None  # compute the difference in frames


def preprocess(image):
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by a factor of 2
    image[image == 144] = 0  # erase background type 1
    image[image == 109] = 0  # erase background type 2
    image[image != 0] = 1  # everything else (paddles, ball) is set to 1
    return image.astype(np.float64).ravel()

# Main loop
while True:
    current = preprocess(observation)
    x = current - previous if previous is not None else np.zeros(DIMENSIONS)
    previous = current
    action = 0
    observation, reward, done, info = env.step(action)
