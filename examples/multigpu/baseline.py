from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
from helpers import get_gpu, train, evaluate

torch.manual_seed(0)

DATASET_DIR = Path(__file__).parent.parent / "datasets"


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


if __name__ == "__main__":
    device = get_gpu()
    transforms = Compose([Resize((224, 224)), ToTensor()])

    training_data = CIFAR10(DATASET_DIR, train=True, transform=transforms, download=True)
    test_data = CIFAR10(DATASET_DIR, train=False, transform=transforms, download=True)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, pin_memory=True)

    model = ResNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0005, momentum=0.9)

    for _ in range(5):
        train(model, train_dataloader, optimizer, device=device)
        evaluate(model, test_dataloader, device=device)
