"""
A tensor is an N-dimensional array
"""
import torch
from numpy import ndarray


def print_tensor(description: str, data: ndarray) -> None:
    print(f'{description} = (\n{data}\n)\n')

a = torch.ones(3, 3)
b = torch.eye(3, 3)

print_tensor('a', a.numpy())
print_tensor('b', b.numpy())
print_tensor('a + b', (a + b).numpy())
print_tensor('a * b', (a * b).numpy())  # matrix-matrix product
print_tensor('matmul(a, b)', torch.matmul(a, b).numpy())
