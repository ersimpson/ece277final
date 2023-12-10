from pathlib import Path
import typing as T
import time
from datetime import datetime

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run(
    epochs: int = 10,
    batch_size: int = 10000,
    lr: float = 0.1,
    save_freq: int = 10,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_path = (Path(__file__).parent.parent / "data").absolute()

    training_data = datasets.MNIST(
        root=str(data_path),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_data = datasets.MNIST(
        root=str(data_path),
        train=False,
        download=True,
        transform=transforms.ToTensor(), 
    )

    train_dataloader = DataLoader(training_data, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    model = MNISTModel()
    model.to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch += 1
        timer = timeit()
        with timer:
            train(
                device=device,
                epoch=epoch,
                loader=train_dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            validate(
                device=device,
                epoch=epoch,
                loader=test_dataloader,
                model=model,
                loss_fn=loss_fn,
            )
        print(f"Epoch {epoch} took {timer.elapsed:.2f} seconds")

        if epoch % save_freq == 0:
            print(f"Saving checkpoint for epoch {epoch}...")
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
            )
            print(f"Checkpoint for epoch {epoch} saved.")


class MNISTModel(torch.nn.Module):

    def __init__(self):
        super(MNISTModel, self).__init__()

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1)
        self.linear1 = torch.nn.Linear(in_features=64*14*14, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(-1, 64*14*14)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


def train(
    device: torch.device,
    epoch: int,
    loader: DataLoader,
    model: MNISTModel,
    loss_fn: T.Callable[[torch.Tensor], torch.Tensor],
    optimizer: SGD,
):
    accuracies = Averager()
    losses = Averager()

    model.train()
    for _, (sample, label) in enumerate(loader):
        sample: torch.Tensor = sample
        sample = sample.to(device)
        label: torch.Tensor = label
        label = label.to(device)

        output = model(sample)
        loss: torch.Tensor = loss_fn(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), sample.size(0))
        accuracies.update(accuracy(output, label), sample.size(0))

    print(
        f"Epoch {epoch} => Avg Train Precision: {accuracies.avg:.4f}, "
        + f"Avg Train Error: {1 - accuracies.avg:.4f}, Avg Train Loss: {losses.avg:.4f}"
    )


def validate(
    device: torch.device,
    epoch: int,
    loader: DataLoader,
    model: MNISTModel,
    loss_fn: T.Callable[[torch.Tensor], torch.Tensor],
):
    accuracies = Averager()
    losses = Averager()

    model.eval()
    for _, (sample, label) in enumerate(loader):
        sample: torch.Tensor = sample
        sample = sample.to(device)
        label: torch.Tensor = label
        label = label.to(device)

        prediction: torch.Tensor = model(sample)
        loss: torch.Tensor = loss_fn(prediction, label)

        losses.update(loss.item(), sample.size(0))
        accuracies.update(accuracy(prediction, label), sample.size(0))

    print(
        f"Epoch {epoch} => Avg Test Precision: {accuracies.avg:.4f}, "
        + f"Avg Test Error: {1 - accuracies.avg:.4f}, Avg Test Loss: {losses.avg:.4f}"
    )


def accuracy(predicted: torch.Tensor, labels: torch.Tensor) -> float: 
    _, predicted_labels = torch.max(predicted, 1)
    correct_predictions = (predicted_labels == labels).sum().item()
    total_samples = labels.size(0)
    accuracy = correct_predictions / total_samples
    return accuracy


def save_checkpoint(
    epoch: int,
    model: MNISTModel,
    optimizer: SGD,
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(state, f"checkpoint_{epoch}_{now}.pt")


class timeit:

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start



class Averager(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count