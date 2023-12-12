import typing as T
import time
from datetime import datetime
import sys

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from py_src.model import MNISTModel


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run(
    data_path: str,
    case: int = 1,
    epochs: int = 10,
    batch_size: int = 10000,
    lr: float = 0.1,
    save_freq: int = 10,
):
    training_data = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_data = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.ToTensor(), 
    )

    train_dataloader = DataLoader(training_data, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # if not torch.cuda.is_available():
    #     print("CUDA is not available. Exiting...")
    #     sys.exit(1)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    model = MNISTModel(case=case)
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
