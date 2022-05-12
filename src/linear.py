"""
Define, train, and evaluate a linear regression model
"""
import torch
from torch import nn, tensor, optim, cat
import matplotlib.pyplot as plt

# Parameters
LR = 0.05
EPOCHS = 50

# Define function to genereate data
f = lambda x: 5 * x + 7

# Create train and test data
x_train = torch.linspace(-5, 5, 100).reshape(-1, 1)
y_train = f(x_train)
x_test = torch.linspace(5, 8, 30).reshape(-1, 1)
y_test = f(x_test)

# Define model
class Net(nn.Module):
    """
    A network for linear regression
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, data: tensor) -> tensor:
        """
        Forward pass
        """
        data = self.linear(data)
        return data


net = Net()

# Define loss and optimization functions
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LR)

# Train the model
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = net(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f'epoch {epoch} loss {loss.item()}')

# Evaluate the model
with torch.no_grad():
    train_output = net(x_train)
    test_output = net(x_test)

# Plot the results
plt.plot(x_train, y_train, 'go', label='x_train', alpha=0.2)
plt.plot(x_test, y_test, 'ro', label='y_train', alpha=0.2)
plt.plot(
    cat([x_train, x_test]),
    cat([train_output, test_output]),
    '--',
    label='predicted',
    alpha=1.0,
)

# Show plot with legend
plt.legend()
plt.show()
