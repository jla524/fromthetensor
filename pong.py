"""
Train a model to play pong
"""
import os

import gym
import numpy as np
from torch import nn, Tensor

# Hyperparameters
GAMMA = 0.99
RENDER = os.getenv('RENDER') is not None
RESUME = os.getenv('RESUME') is not None

# Model initialization
NUM_HIDDEN = 200
DIMENSIONS = 80 * 80


class PongNet(nn.Module):
    """
    A neural net to play pong
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(DIMENSIONS, NUM_HIDDEN)
        self.layer2 = nn.Linear(NUM_HIDDEN, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, data):
        """
        Forward pass
        """
        data = self.layer1(data)
        data = self.act1(data)
        data = self.layer2(data)
        data = self.act2(data)
        return data

model = PongNet()

# Game related
if RENDER:
    env = gym.make('ALE/Pong-v5', render_mode='human')
else:
    env = gym.make('ALE/Pong-v5')


def preprocess(image: np.ndarray) -> np.array:
    """
    Preprocess a 210x160x3 frame into a 6400 (80x80) float vector
    :param image: a raw frame from pong
    :return: a processed frame
    """
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by a factor of 2
    image[image == 144] = 0  # erase background type 1
    image[image == 109] = 0  # erase background type 2
    image[image != 0] = 1  # everything else (paddles, ball) is set to 1
    return image.astype(np.float64).ravel()


def discount_rewards(rewards: np.array) -> np.array:
    """
    Take 1D float array of rewards and compute discounted rewards
    :param rewards: an array of rewards
    :return: an array of discounted rewards
    """
    discounted = np.zeros_like(rewards)
    running_add = 0
    for i in reversed(range(rewards.size)):
        if rewards[i] != 0:
            running_add = 0  # reset the sum since this was a game boundary
        running_add = running_add * GAMMA + rewards[i]
        discounted[i] = running_add
    return discounted


def train() -> None:
    """
    Train the neural net to play pong
    :return: None
    """
    observation = env.reset()
    previous = None
    rewards = []

    while True:
        # Preprocess the observation
        current = preprocess(observation)
        frame = (current - previous if previous is not None
                 else np.zeros(DIMENSIONS))
        previous = current

        # Forward the policy network
        prob = model.forward(Tensor(frame))
        action = 2 if np.random.uniform() < prob else 3

        # Step the environment
        observation, reward, done, info = env.step(action)

        rewards.append(reward)

        if done:
            episode_reward = np.vstack(rewards)
            rewards = []  # reset list memory

            # compute and standardize the discounted reward
            discounted_episode_reward = discount_rewards(episode_reward)
            discounted_episode_reward -= np.mean(discounted_episode_reward)
            discounted_episode_reward /= np.std(discounted_episode_reward)
            print(discounted_episode_reward)


if __name__ == '__main__':
    train()
