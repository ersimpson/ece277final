import sys
from pathlib import Path
import typing as T
import time

CUSTOM_PYTORCH_ROOT_PATH = (Path(__file__).parent.parent / "build" / "src" / "cuda_model").absolute()
CUSTOM_PYTORCH_DEBUG_PATH = (CUSTOM_PYTORCH_ROOT_PATH / "Debug").absolute()
CUSTOM_PYTORCH_RELEASE_PATH = (CUSTOM_PYTORCH_ROOT_PATH / "Release").absolute()
if CUSTOM_PYTORCH_DEBUG_PATH.exists() and CUSTOM_PYTORCH_DEBUG_PATH not in sys.path:
    sys.path.append(str(CUSTOM_PYTORCH_DEBUG_PATH))
if CUSTOM_PYTORCH_RELEASE_PATH.exists() and CUSTOM_PYTORCH_RELEASE_PATH not in sys.path:
    sys.path.append(str(CUSTOM_PYTORCH_RELEASE_PATH))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import mnist_cpp_model
except ImportError:
    print("Could not import mnist_cpp_model. Exiting...")
    sys.exit(1)


def set_seed(seed: int = 1):
    np.random.seed(seed)


def run(
    data_path: str,
    case: int = 1,
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

    model = TwoLayerNN(input_size=784, hidden_size=128, output_size=10, case=case)
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


class MNISTNumpyTransform:
    
    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        sample: np.ndarray = sample.numpy()
        sample = sample.reshape(-1, 28*28)
        return sample


class TwoLayerNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, case: int = 1):
        self.case = case
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = self.init_weights(self.input_size, self.hidden_size)
        self.b1 = self.init_bias(self.hidden_size)
        self.w2 = self.init_weights(self.hidden_size, self.output_size)
        self.b2 = self.init_bias(self.output_size)

    def init_weights(self, in_size: int, out_size: int) -> np.ndarray:
        return np.random.randn(in_size, out_size) * 0.01

    def init_bias(self, size: int) -> np.ndarray:
        return np.zeros((1, size))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x / np.sum(x, axis=1, keepdims=True)

    def mm(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        #if self.case == 2:
        #    N, _ = x.shape
        #    _, M = y.shape
        #    return mnist_cpp_model.mm(x, y).reshape(N, M)
        return np.dot(x, y)
    
    def madd(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.case == 2:
            N, M = x.shape
            return mnist_cpp_model.madd(x, y).reshape(N, M)
        return x + y
    
    def mt(self, x: np.ndarray) -> np.ndarray:
        if self.case == 2:
            N, M = x.shape
            return mnist_cpp_model.mt(x).reshape(M, N)
        return x.T
    
    def mmelem(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.case == 2:
            N, M = x.shape
            return mnist_cpp_model.mmelem(x, y).reshape(N, M)
        return x * y
    
    def mmreduce(self, x: np.ndarray) -> np.ndarray:
        if self.case == 2:
            _, M = x.shape
            return mnist_cpp_model.mmreduce(x).reshape(1, M)
        return np.sum(x, axis=0, keepdims=True)
    
    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        N, _ = inputs.shape
        b1 = np.vstack((N * [self.b1]))
        z1 = self.madd(self.mm(inputs, self.w1), b1)
        self.a1 = self.sigmoid(z1)
        N, _ = self.a1.shape
        b2 = np.vstack((N * [self.b2]))
        z2 = self.madd(self.mm(self.a1, self.w2), b2)
        self.out = self.softmax(z2)        
        return self.out

    def backward_pass(self, inputs: np.ndarray, targets: np.ndarray, lr: float):
        N, _ = inputs.shape
        l2_delta = (self.out - targets) / N

        grad_w2 = self.mm(self.mt(self.a1), l2_delta)
        grad_b2 = self.mmreduce(l2_delta)

        d_sigmoid = self.mmelem(self.a1, 1 - self.a1)
        l1_delta = self.mmelem(self.mm(l2_delta, self.mt(self.w2)), d_sigmoid) 
        grad_w1 = self.mm(self.mt(inputs), l1_delta)
        grad_b1 = self.mmreduce(l1_delta)

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


def reshape_targets_to_numpy(tensor: torch.Tensor, C: int = 10) -> np.ndarray:
    v: np.ndarray = tensor.numpy()
    N = v.size
    one_hot = np.eye(C)[v.tolist(), :].reshape(N, C)
    return one_hot


def loss_fn(targets: np.ndarray, outputs: np.ndarray) -> float:
    N, _ = outputs.shape
    return -np.sum(targets * np.log(outputs + 1e-8)) / N


def accuracy(outputs: np.ndarray, targets: np.ndarray) -> float:
    N, _ = outputs.shape
    digits = np.argmax(outputs, axis=1).reshape(N, 1)
    targets = np.argmax(targets, axis=1).reshape(N, 1)
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
