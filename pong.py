"""
Train a model to play pong
"""
import os

import gym
import numpy as np
import torch
from torch import optim, nn, Tensor
from torch.autograd import Variable

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
RENDER = os.getenv('RENDER') is not None
RESUME = os.getenv('RESUME') is not None

# Model initialization
NUM_HIDDEN = 200
DIMENSIONS = 80 * 80
MODEL_PATH = 'save.p'


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

    def forward(self, data: Tensor) -> Tensor:
        """
        Forward pass
        :param data: a frame in the pong game
        :return: a probability to move the paddle up
        """
        data = self.layer1(data)
        data = self.act1(data)
        data = self.layer2(data)
        data = self.act2(data)
        return data

if RESUME:
    model = torch.load(MODEL_PATH)
    model.eval()
else:
    model = PongNet()
optimizer = optim.Adam(model.parameters(), lr=LR)

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
    rewards, losses = [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    while True:
        # Preprocess the observation
        current = preprocess(observation)
        frame = (current - previous if previous is not None
                 else np.zeros(DIMENSIONS))
        previous = current
        # Forward the policy network
        prob = model.forward(Tensor(frame))
        action = 2 if np.random.uniform() < prob else 3
        # Record intermediates needed for backprop
        label = 1 if action == 2 else 0
        losses.append(label - prob.item())
        # Step the environment
        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        rewards.append(reward)

        if done:  # An episode finished
            episode_number += 1
            episode_loss = np.vstack(losses)
            episode_reward = np.vstack(rewards)
            # Compute and standardize the discounted reward
            episode_reward = discount_rewards(episode_reward)
            episode_reward -= np.mean(episode_reward)
            episode_reward /= np.std(episode_reward)
            # Backward pass
            episode_loss *= episode_reward
            loss = Variable(Tensor(episode_loss), requires_grad=True).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Book-keeping
            running_reward = (reward_sum if running_reward is None
                              else running_reward * 0.99 + reward_sum * 0.01)
            print(f'episode reward total {reward_sum} '
                  f'running reward {running_reward}')
            if episode_number % 100 == 0:
                torch.save(model, MODEL_PATH)
            # Reset variables
            observation = env.reset()
            previous = None
            rewards, losses = [], []
            reward_sum = 0


if __name__ == '__main__':
    train()
