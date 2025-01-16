import numpy as np
from PIL import Image
import torch
from torch import nn
from datasets import load_dataset
from helpers import get_gpu, train, evaluate

torch.manual_seed(0)
device = get_gpu()


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def __call__(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out = (out + x).relu()
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def __call__(self, x):
        x = self.bn1(self.conv1(x)).relu()
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


def transform(x):
    x = x.reshape(-1, 1, 32, 32)
    x = [[Image.fromarray(z).resize((224, 224)) for z in y] for y in x]
    x = np.stack([np.stack([np.asarray(z) for z in y], axis=0) for y in x], axis=0)
    x = x.reshape(-1, 3, 224, 224)
    return x


if __name__ == "__main__":
    dataset = load_dataset("cifar10")

    X_train = (np.array([np.array(image) for image in dataset["train"]["img"]]) / 255.0 - 0.5) / 0.25
    Y_train = np.array(dataset["train"]["label"], dtype=np.int32)
    X_test = (np.array([np.array(image) for image in dataset["test"]["img"]]) / 255.0 - 0.5) / 0.25
    Y_test = np.array(dataset["test"]["label"], dtype=np.int32)

    model = ResNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0005, momentum=0.9)
    train(model, X_train, Y_train, optimizer, 50000, BS=16, transform=transform, device=device)
    evaluate(model, X_test, Y_test, transform=transform, device=device)
