"""
Train a neural network to recognize handwritten digits
"""
from hashlib import sha224
from pathlib import Path
from gzip import decompress

import requests
import numpy as np
from torch import nn, tensor, flatten, optim

# Parameters
LR = 0.01
EPOCHS = 1000

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
        #self.conv = nn.Conv2d(28, 28, (3, 3))
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, data: tensor) -> tensor:
        """
        Forward pass
        :param data: an image to pass in
        :return: a vector of probabilities
        """
        data = data.view(-1, 28 * 28)
        #data = self.conv(data)
        #data = flatten(data)
        data = self.linear(data)
        return data

net = Net()
optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.NLLLoss()

# Train the network
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = net.forward(tensor(x_train, dtype=float))
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    print(f'epoch {epoch} loss {loss}')
