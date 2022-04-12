"""
A tensor is an N-dimensional array
"""
import torch
from numpy import ndarray


def print_tensor(description: str, tensor: torch.tensor) -> None:
    """
    Print the description and the data in the given tensor
    :param description: a description of the tensor
    :param tensor: a tensor initialized with PyTorch
    :return: None
    """
    print(f'{description} = (\n{tensor.numpy()}\n)\n')


a = torch.ones(3, 3)
b = torch.eye(3, 3)

print_tensor('a', a)
print_tensor('b', b)
print_tensor('a + b', a + b)
print_tensor('a @ b', a @ b)  # matrix multiplication
print_tensor('a * b', a * b)  # matrix-matrix product
