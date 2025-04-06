import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F


np.random.seed(0)


def get_device() -> str:
    backends = {torch.cuda: "cuda", torch.backends.mps: "mps"}
    for backend, name in backends.items():
        if backend.is_available():
            return name
    return "cpu"


def train(
    model,
    X_train,
    Y_train,
    optim,
    steps,
    batch_size=128,
    lossfn=F.cross_entropy,
    transform=lambda x: x,
    device="cpu",
):
    model.train()
    for i in (t := trange(steps)):
        sample = np.random.randint(0, X_train.shape[0], size=batch_size)
        x = torch.tensor(transform(X_train[sample]), device=device)
        y = torch.tensor(Y_train[sample], device=device)
        output = model(x)
        loss = lossfn(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        cat = output.argmax(dim=-1)
        accuracy = (cat == y).float().mean()
        t.set_description(f"loss {loss.item():.2f} accuracy {accuracy:.2f}")


def evaluate(
    model,
    X_test,
    Y_test,
    batch_size=128,
    target_transform=lambda x: x,
    device="cpu",
):
    model.eval()
    preds = np.zeros(Y_test.shape)
    for i in trange((len(Y_test)-1)//batch_size+1):
        x = target_transform(X_test[i*batch_size:(i+1)*batch_size])
        y = model(torch.tensor(x, device=device, requires_grad=False))
        preds[i*batch_size:(i+1)*batch_size] = y.argmax(dim=-1).detach().cpu().numpy()
    accuracy = (Y_test == preds).mean()
    print(f"test set accuracy is {accuracy}")
