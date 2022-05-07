"""
Train an agent to play pong with neural net
"""
import os
from pathlib import Path

import gym
import numpy as np
import torch
from torch import optim, nn, Tensor

# Hyperparameters
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 10
DISCOUNT_FACTOR = 0.99
RENDER = os.getenv('RENDER') is not None
RESUME = os.getenv('RESUME') is not None

# Model initialization
DIMENSIONS = 80 * 80
NUM_HIDDEN = 400
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
criterion = nn.BCELoss(reduction='none')

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
        self.labels = []
        self.running_reward = None
        self.episode_number = 0

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
        result -= np.mean(result)
        result /= np.std(result)
        return result

    def _update_parameters(self) -> None:
        """
        Perform a backward pass and an optimzation step
        :return: None
        """
        predicted = torch.cat(self.probs).float()
        expected = torch.tensor(self.labels).float()
        episode_reward = self._get_episode_reward()
        reward_tensor = torch.from_numpy(episode_reward).float().squeeze()

        losses = criterion(predicted, expected) * reward_tensor
        loss = torch.mean(losses)
        loss.backward()

        if self.episode_number % BATCH_SIZE == 0:
            optimizer.step()
            optimizer.zero_grad()

    def _print_rewards(self) -> None:
        """
        Compute and print the total and running reward
        :return: None
        """
        reward_sum = sum(self.rewards)

        if self.running_reward is None:
            self.running_reward = reward_sum
        else:
            running = self.running_reward * DISCOUNT_FACTOR
            current = reward_sum * (1 - DISCOUNT_FACTOR)
            self.running_reward = running + current

        print(f'episode reward total {reward_sum} '
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
        self.labels = []

    def train(self) -> None:
        """
        Train the neural net to play pong
        :return: None
        """
        model.train()

        while True:
            frame = self._get_frame()
            prob = model.forward(Tensor(frame))
            action = 2 if np.random.uniform() < prob.item() else 3
            label = 1 if action == 2 else 0
            self.probs.append(prob)
            self.labels.append(label)

            self.observation, reward, done, _ = env.step(action)
            self.rewards.append(reward)

            if done:  # An episode finished
                self.episode_number += 1
                self._update_parameters()
                self._print_rewards()
                if self.episode_number % 100 == 0:
                    torch.save(model, SAVE_PATH)
                self.reset_variables()


if __name__ == '__main__':
    agent = PongAgent()
    agent.train()
