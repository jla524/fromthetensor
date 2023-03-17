import torch
import numpy as np
from tqdm import trange
import torch.nn.functional as F
np.random.seed(1337)


def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=F.cross_entropy, transform=lambda x: x):
    model.train()
    for i in (t := trange(steps)):
        sample = np.random.randint(0, X_train.shape[0], size=(BS))
        x = torch.tensor(transform(X_train[sample]), requires_grad=False)
        y = torch.tensor(Y_train[sample])
        output = model(x)
        loss = lossfn(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        cat = output.argmax(dim=-1)
        accuracy = (cat == y).float().mean()
        loss = loss.item()
        t.set_description(f"loss {loss:.2f} accuracy {accuracy:.2f}")


def evaluate(model, X_test, Y_test, num_classes=10, BS=128, return_predict=False, transform=lambda x: x):
    model.eval()
    outputs = np.zeros(list(Y_test.shape) + [num_classes])
    for i in trange((len(Y_test)-1)//BS+1):
        x = torch.tensor(transform(X_test[i*BS:(i+1)*BS]))
        outputs[i*BS:(i+1)*BS] = model(x).detach().numpy()
    preds = outputs.argmax(axis=-1)
    accuracy = (Y_test == preds).mean()
    print(f"test set accuracy is {accuracy}")
    return (accuracy, preds) if return_predict else accuracy
