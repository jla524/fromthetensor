"""
Train a neural network to recogniz handwritten digits
"""
from hashlib import sha224
from pathlib import Path
from gzip import decompress

import requests
import numpy as np


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


x_train = fetch_dataset('train-images-idx3-ubyte.gz')[16:]
y_train = fetch_dataset('train-labels-idx1-ubyte.gz')[8:]
x_test = fetch_dataset('t10k-images-idx3-ubyte.gz')[16:]
y_test = fetch_dataset('t10k-labels-idx1-ubyte.gz')[8:]

print(x_train)
