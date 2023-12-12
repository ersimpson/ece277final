import typing as T
import time

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

np.set_printoptions(threshold=np.inf)

def set_seed(seed: int = 1):
    np.random.seed(seed)


def run(
    data_path: str,
    epochs: int = 10,
    batch_size: int = 10,
    lr: float = 0.1,
    batches: T.Optional[int] = None,
):
    training_data = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=MNISTNumpyTransform(),
    )
    test_data = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=MNISTNumpyTransform(), 
    )

    train_dataloader = DataLoader(training_data, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    model = TwoLayerNN(input_size=784, hidden_size=64, output_size=10)
    for epoch in range(epochs):
        epoch += 1
        timer = timeit()
        with timer:
            train(
                loader=train_dataloader,
                epoch=epoch,
                model=model,
                lr=lr,
                batches=batches,
            )
            validate(
                loader=test_dataloader,
                epoch=epoch,
                model=model,
                lr=lr,
                batches=batches,
            )
        print(f"Epoch {epoch} took {timer.elapsed:.2f} seconds")


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def d_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return x / np.sum(x, axis=1, keepdims=True)


class MNISTNumpyTransform:
    
    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        sample: np.ndarray = sample.numpy()
        sample = sample.reshape(-1, 28*28)
        return sample


class TwoLayerNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        self.a1 = relu(np.dot(inputs, self.w1) + self.b1)
        self.out = softmax(np.dot(self.a1, self.w2) + self.b2)        
        return self.out

    def backward_pass(self, inputs: np.ndarray, targets: np.ndarray, lr: float):
        N, C = self.out.shape
        one_hot = np.eye(C)[targets].reshape(N, C)
        l2_delta = (self.out - one_hot) / N

        grad_w2 = np.dot(self.a1.T, l2_delta)
        grad_b2 = np.sum(l2_delta, axis=0, keepdims=True)

        l1_delta = d_relu(np.dot(l2_delta, self.w2.T))
        grad_w1 = np.dot(inputs.T, l1_delta)
        grad_b1 = np.sum(l1_delta, axis=0, keepdims=True)

        self.w1 -= lr * grad_w1
        self.b1 -= lr * grad_b1
        self.w2 -= lr * grad_w2
        self.b2 -= lr * grad_b2


def train(
    loader: DataLoader,
    epoch: int,
    model: TwoLayerNN,
    lr: float,
    batches: T.Optional[int] = None,
    validate: bool = False,
):
    accuracies = Averager()
    losses = Averager()

    for batch, (inputs, targets) in enumerate(loader):
        inputs = reshape_inputs_to_numpy(inputs)
        targets = reshape_targets_to_numpy(targets)

        outputs = model.forward_pass(inputs)
        if not validate:
            model.backward_pass(inputs, targets, lr)
        loss = loss_fn(targets, outputs)

        accuracies.update(accuracy(outputs, targets), outputs.shape[0])
        losses.update(loss, outputs.shape[0])

        if batches is None:
            continue
        if batch+1 >= batches:
            break
    
    train_mode = "Train" if not validate else "Validation"
    print(
        f"Epoch {epoch} => Avg {train_mode} Precision: {accuracies.avg:.4f}, "
        + f"Avg {train_mode} Error: {1 - accuracies.avg:.4f}, Avg {train_mode} Loss: {losses.avg:.4f}"
    )


def validate(
    loader: DataLoader,
    epoch: int,
    model: TwoLayerNN,
    lr: float,
    batches: T.Optional[int] = None,
):
    train(
        loader=loader,
        epoch=epoch,
        model=model,
        lr=lr,
        batches=batches,
        validate=True,
    )


def reshape_inputs_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    v: np.ndarray = tensor.numpy()
    N, _, M = v.shape
    return v.reshape(N, M)

def reshape_targets_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    v: np.ndarray = tensor.numpy()
    N = v.size
    return v.reshape(N, 1)

def loss_fn(targets: np.ndarray, outputs: np.ndarray) -> float:
    N, C = outputs.shape
    one_hot = np.eye(C)[targets].reshape(N, C)
    return -np.sum(one_hot * np.log(outputs + 1e-8))

def accuracy(outputs: np.ndarray, targets: np.ndarray) -> float:
    N, _ = outputs.shape
    digits = np.argmax(outputs, axis=1).reshape(N, 1)
    return np.mean(digits == targets)

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