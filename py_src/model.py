import sys
from pathlib import Path
from typing import Any, Tuple

CUSTOM_PYTORCH_ROOT_PATH = (Path(__file__).parent.parent / "build" / "src" / "cuda_model").absolute()
CUSTOM_PYTORCH_DEBUG_PATH = (CUSTOM_PYTORCH_ROOT_PATH / "Debug").absolute()
CUSTOM_PYTORCH_RELEASE_PATH = (CUSTOM_PYTORCH_ROOT_PATH / "Release").absolute()
if CUSTOM_PYTORCH_DEBUG_PATH.exists() and CUSTOM_PYTORCH_DEBUG_PATH not in sys.path:
    sys.path.append(str(CUSTOM_PYTORCH_DEBUG_PATH))
if CUSTOM_PYTORCH_RELEASE_PATH.exists() and CUSTOM_PYTORCH_RELEASE_PATH not in sys.path:
    sys.path.append(str(CUSTOM_PYTORCH_RELEASE_PATH))

import torch
import numpy as np

try:
    import mnist_cpp_model
except ImportError:
    print("Could not import mnist_cpp_model. Exiting...")
    sys.exit(1)


class MNISTModel(torch.nn.Module):

    # case 1 - run model using PyTorch CUDA implementation
    # case 2 - run model using PyTorch with C++ extensions
    # case 3 - run model using PyTorch with C++/CUDA extensions
    # case 4 - run model using PyTorch with C++/CUDA extensions and performance optimizations

    def __init__(self, case: int = 1):
        super(MNISTModel, self).__init__()
        self.case = case

        if self.case == 1:
            self.relu = torch.nn.ReLU()
            self.softmax = torch.nn.Softmax(dim=1)
            self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1)
            self.linear1 = torch.nn.Linear(in_features=64*14*14, out_features=128)
            self.linear2 = torch.nn.Linear(in_features=128, out_features=10)

        elif self.case == 2:
            self.relu = MyReLU()
            self.softmax = MySoftmax()
            self.pool = MyAvgPool2d(kernel=(2, 2), stride=2)
            self.conv1 = MyConv2D(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
            self.linear1 = MyLinear(input_features=64*14*14, output_features=128)
            self.linear2 = MyLinear(input_features=128, output_features=10)

        raise NotImplementedError("Only case 1 & 2 are implemented")

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


class MyConv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(input: np.ndarray, kernel: np.ndarray, stride: int, padding: int):
        return torch.tensor(conv2d(input, kernel, stride, padding))
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        input, kernel, stride, padding = inputs
        ctx.save_for_backward(input, kernel, stride, padding)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input, kernel, stride, padding = ctx.saved_tensors
        rotated_kernel = np.rot90(kernel, k=2)
        grad_input = conv2d(rotated_kernel, grad_output, stride, padding)
        grad_kernel = conv2d(input, grad_output, stride, padding)
        return torch.tensor(grad_input), torch.tensor(grad_kernel)


class MyConv2D(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: int, padding: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        torch.nn.init.uniform_(self.kernel, -0.1, 0.1)

    def forward(self, input: torch.Tensor):
        input = input.numpy()
        kernel = self.kernel.numpy()
        return MyConv2dFunction.apply(input, kernel, self.stride, self.padding)


class MyLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(input: np.ndarray, weight: np.ndarray, bias: np.ndarray):
        return torch.tensor(np.concatenate(input, 1) @ np.vstack((weight.T, bias.T)))

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input, weights, _ = ctx.saved_tensors
        grad_input = grad_output @ weights
        grad_weights = grad_output.T @ input
        grad_bias = grad_output.sum(axis=0)
        return torch.tensor(grad_input), torch.tensor(grad_weights), torch.tensor(grad_bias)


class MyLinear(torch.nn.Module):

    def __init__(self, input_features, output_features):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = torch.nn.Parameter(torch.empty(output_features, input_features))
        self.bias = torch.nn.Parameter(torch.empty(output_features))

        torch.nn.init.uniform_(self.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input: torch.Tensor):
        input = input.numpy()
        weight = self.weight.numpy()
        bias = self.bias.numpy()
        return MyLinearFunction.apply(input, weight, bias)


class MyAvgPool2d(torch.autograd.Function):

    @staticmethod
    def forward(input: torch.Tensor, kernel: Tuple[int, int], stride: int):
        input = input.numpy()
        return torch.tensor(conv2d(input, np.ones(kernel), stride, 0, func=np.mean))

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, kernel, stride = inputs
        ctx.save_for_backward(input, kernel, stride)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.numpy()
        _, kernel, _ = ctx.saved_tensors
        return torch.tensor((1 / kernel.numel()) * grad_output)


class MySoftmax(torch.autograd.Function):

    @staticmethod
    def forward(input: torch.Tensor):
        input = input.numpy()
        return torch.tensor(softmax(input))
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.numpy()
        output = ctx.saved_tensors
        return torch.tensor(np.multiply(grad_output, d_softmax(output)))
    

class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(input: torch.Tensor):
        input = input.numpy()
        return torch.tensor(relu(input))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.numpy()
        inputs = ctx.saved_tensors
        return torch.tensor(np.multiply(grad_output, d_relu(inputs)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=0)
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp, axis=0)
    

def d_softmax(z: np.ndarray) -> np.ndarray:
    grad = np.vstack(z * z.shape[0])
    N, M = grad.shape
    for i in range(N):
        for j in range(M):
            if i == j:
                grad[i, j] = grad[i, j] * (1 - grad[i, j])
            else:
                grad[i, j] = -z[i] * grad[i, j]
    return grad


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)


def d_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0)


def conv2d(x: np.ndarray, k: np.ndarray, stride: int, padding: int, func = np.sum) -> np.ndarray:
    N, H, W = x.shape
    F, HH, WW = k.shape
    H_out = (H + 2 * padding - HH) // stride + 1
    W_out = (W + 2 * padding - WW) // stride + 1
    out = np.zeros((N, F, H_out, W_out))
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant", constant_values=0)
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, f, i, j] = func(x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * k[f, :])
    return out