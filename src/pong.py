"""
Train an agent to play pong with neural net
"""
import os
from pathlib import Path

import gym
import numpy as np
import torch
from torch import optim, nn, Tensor
from torch.autograd import Variable

# Hyperparameters
LR = 1e-3
GAMMA = 0.99
DISCOUNT_FACTOR = 0.99
RENDER = os.getenv('RENDER') is not None
RESUME = os.getenv('RESUME') is not None

# Model initialization
DIMENSIONS = 80 * 80
NUM_HIDDEN = 200
NUM_OUTPUT = 1
BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_PATH / 'models'
SAVE_PATH = MODEL_PATH / 'save.pt'


class PongNet(nn.Module):
    """
    A neural net to play pong
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(DIMENSIONS, NUM_HIDDEN, bias=False)
        self.layer2 = nn.Linear(NUM_HIDDEN, NUM_OUTPUT, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data: Tensor) -> Tensor:
        """
        Forward pass
        :param data: a frame in the pong game
        :return: a probability to move the paddle up
        """
        data = self.layer1(data)
        data = self.relu(data)
        data = self.layer2(data)
        data = self.sigmoid(data)
        return data

if RESUME and os.path.isfile(SAVE_PATH):
    model = torch.load(SAVE_PATH)
else:
    MODEL_PATH.mkdir(exist_ok=True)
    model = PongNet()
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()

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


class PongAgent:
    """
    An agent to play pong
    """
    def __init__(self):
        self.observation = env.reset()
        self.previous = None
        self.rewards = []
        self.probs = []
        self.losses = []
        self.reward_sum = 0
        self.running_reward = None

    def _get_frame(self) -> np.array:
        """
        Compute the difference between the current and previous processed frame
        :return: a frame to feed into the neural net
        """
        processed = preprocess(self.observation)
        frame = (processed - self.previous
                 if self.previous is not None
                 else np.zeros(DIMENSIONS))
        self.previous = processed
        return frame

    def _get_episode_reward(self) -> np.ndarray:
        """
        Compute the standardized discounted reward
        :return: an episode reward
        """
        result = np.vstack(self.rewards)
        result = discount_rewards(result)
        #result -= np.mean(result)
        #result /= np.std(result)
        return result

    def _update_parameters(self) -> None:
        """
        Perform a backward pass and an optimzation step
        """
        episode_reward = self._get_episode_reward()
        expected = episode_reward * np.vstack(self.losses)
        predicted = torch.vstack(self.probs)
        optimizer.zero_grad()
        loss = criterion(predicted, Tensor(expected))
        loss.backward()
        optimizer.step()

    def _print_rewards(self) -> None:
        """
        Compute and print the total and running reward
        :return: None
        """
        if self.running_reward is None:
            self.running_reward = self.reward_sum
        else:
            running = self.running_reward * DISCOUNT_FACTOR
            current = self.reward_sum * (1 - DISCOUNT_FACTOR)
            self.running_reward = running + current

        print(f'episode reward total {self.reward_sum} '
              f'running reward {self.running_reward}')

    def reset_variables(self) -> None:
        """
        Reset variables
        :return: None
        """
        self.observation = env.reset()
        self.previous = None
        self.rewards = []
        self.probs = []
        self.losses = []
        self.reward_sum = 0

    def train(self) -> None:
        """
        Train the neural net to play pong
        :return: None
        """
        episode_number = 0
        model.train()

        while True:
            frame = self._get_frame()
            prob = model.forward(Tensor(frame))
            action = 2 if np.random.uniform() < prob else 3
            label = 1 if action == 2 else 0
            self.probs.append(prob)
            self.losses.append(label)

            self.observation, reward, done, _ = env.step(action)
            self.reward_sum += reward
            self.rewards.append(reward)

            if done:  # An episode finished
                episode_number += 1
                self._update_parameters()
                self._print_rewards()
                if episode_number % 100 == 0:
                    torch.save(model, SAVE_PATH)
                self.reset_variables()


if __name__ == '__main__':
    agent = PongAgent()
    agent.train()
