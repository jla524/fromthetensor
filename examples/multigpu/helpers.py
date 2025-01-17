import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

np.random.seed(1337)


def get_gpu():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(model, dataloader, optim, lossfn=F.cross_entropy, device="cpu"):
    model.train()
    for x, y in (t := tqdm(dataloader, total=len(dataloader))):
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        cat = out.argmax(dim=-1)
        accuracy = (cat == y).float().mean()
        t.set_description(f"loss {loss.item():.2f} accuracy {accuracy:.2f}")


def evaluate(model, dataloader, device="cpu"):
    correct = 0
    with torch.no_grad():
        model.eval()
        for x, y in tqdm(dataloader, total=len(dataloader)):
            x = x.to(device)
            y = y.to(device)
            pred = torch.argmax(model(x), dim=-1)
            correct += (pred == y).sum()
    accuracy = correct / len(dataloader.dataset)
    print(f"test set accuracy is {accuracy}")
