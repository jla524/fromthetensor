import torch
import numpy as np
from tqdm import trange
import torch.nn.functional as F
np.random.seed(1337)


def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=F.cross_entropy):
    model.train()
    for i in (t := trange(steps)):
        sample = np.random.randint(0, high=X_train.shape[0], size=(BS))
        x = torch.tensor(X_train[sample])
        y = torch.tensor(Y_train[sample])
        out = model(x)
        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        cat = torch.argmax(out, dim=1)
        accuracy = (cat == y).float().mean()
        loss = loss.item()
        t.set_description(f"loss {loss:.2f} accuracy {accuracy:.2f}")


def evaluate(model, X_test, Y_test):
    model.eval()
    out = model(torch.tensor(X_test))
    preds = torch.argmax(out, dim=1).numpy()
    accuracy = (Y_test == preds).mean()
    print(f"test set accuracy is {accuracy}")
