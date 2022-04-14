"""
Train a neural network to recogniz handwritten digits
"""
from hashlib import sha224
from pathlib import Path

import numpy as np
import requests


def fetch_dataset(file_name: str) -> np.array:
    file_path = Path('/tmp') / file_name
    if file_path.exists():
        buffer = file_path.open('r')
    else:
        url = f'http://yann.lecun.com/exdb/mnist/{file_name}'
        buffer = requests.get('url')
    return np.frombuffer(buffer)


x_train = fetch_dataset('train-images-idx3-ubyte.gz')
y_train = fetch_dataset('train-labels-idx1-ubytes.gz')
x_test = fetch_dataset('t10k-images-idx3-ubyte.gz')
y_test = fetch_dataset('t10k-labels-idx1-ubytes.gz')
