import torch
import numpy as np
from tqdm import trange
import torch.nn.functional as F
np.random.seed(1337)


def get_gpu():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=F.cross_entropy, transform=lambda x: x, device=torch.device("cpu")):
    model.train()
    for i in (t := trange(steps)):
        sample = np.random.randint(0, X_train.shape[0], size=(BS))
        x = torch.tensor(transform(X_train[sample]), requires_grad=False).to(device)
        y = torch.tensor(Y_train[sample]).to(device)
        out = model(x)
        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        cat = output.argmax(dim=-1)
        accuracy = (cat == y).float().mean()
        t.set_description(f"loss {loss.item():.2f} accuracy {accuracy:.2f}")


def evaluate(model, X_test, Y_test, BS=128, transform=lambda x: x, device=torch.device("cpu")):
    model.eval()
    preds = np.zeros(Y_test.shape)
    for i in trange((len(Y_test)-1)//BS+1):
        out = model(torch.tensor(transform(X_test[i*BS:(i+1)*BS])).to(device))
        preds[i*BS:(i+1)*BS] = torch.argmax(out, dim=-1).detach().cpu().numpy()
    accuracy = (Y_test == preds).mean()
    print(f"test set accuracy is {accuracy}")
