"""
Train a neural network to recognize handwritten digits
"""
from hashlib import sha224
from pathlib import Path
from gzip import decompress

import requests
import numpy as np
import torch
from torch import nn, tensor, optim

# Parameters
BS = 128
LR = 0.0002
EPOCHS = 10000

# https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb
def fetch_dataset(file_name: str) -> np.array:
    """
    Load the given file from the MNIST database
    :file_name: a file in the MNIST database
    """
    file_path = Path('/tmp') / sha224(file_name.encode('utf-8')).hexdigest()
    if file_path.exists():
        with file_path.open('rb') as file:
            data = file.read()
    else:
        url = f'http://yann.lecun.com/exdb/mnist/{file_name}'
        data = requests.get(url).content
        with file_path.open('wb') as file:
            file.write(data)
    return np.frombuffer(decompress(data), dtype=np.uint8).copy()


# Create the training and testing set
x_train = fetch_dataset('train-images-idx3-ubyte.gz')[16:].reshape(-1, 28, 28)
y_train = fetch_dataset('train-labels-idx1-ubyte.gz')[8:]
x_test = fetch_dataset('t10k-images-idx3-ubyte.gz')[16:].reshape(-1, 28, 28)
y_test = fetch_dataset('t10k-labels-idx1-ubyte.gz')[8:]


class Net(nn.Module):
    """
    A neural network to recognize handwritten digits
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 32, bias=False)
        self.layer2 = nn.Linear(32, 10, bias=False)
        self.act = nn.LogSoftmax(dim=1)

    def forward(self, data: tensor) -> tensor:
        """
        Forward pass
        :param data: an image to pass in
        :return: a vector of probabilities
        """
        data = self.layer1(data)
        data = self.layer2(data)
        data = self.act(data)
        return data

net = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

# Train the network
for epoch in range(EPOCHS):
    indices = np.random.randint(0, len(x_train), size=(BS))
    x = tensor(x_train[indices].reshape((-1, 28 * 28))).float()
    y = tensor(y_train[indices]).long()
    net.zero_grad()
    out = net.forward(x)
    loss = criterion(out, y).mean()
    loss.backward()
    optimizer.step()
    print(f'epoch {epoch} loss {loss}')

# Evaluation
with torch.no_grad():
    x = tensor(x_test.reshape((-1, 28 * 28))).float()
    y = tensor(y_test)
    out = net.forward(x)
    pred = out.argmax(dim=1)
    accuracy = (pred == y).float().mean()
    print(f'accuracy {accuracy}')
