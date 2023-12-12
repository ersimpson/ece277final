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

        if self.case not in (1, 2, 3):
            raise NotImplementedError("Only case 1 & 2 are implemented")

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.linear1 = torch.nn.Linear(in_features=784, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=10)

        if self.case == 2:
            self.relu = MyReLU()
            self.softmax = MySoftmax()
            #self.linear1 = MyLinear(input_features=64*14*14, output_features=128)
            #self.linear2 = MyLinear(input_features=128, output_features=10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 784)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class MyLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        return mnist_cpp_model.linear_forward(input, weight, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input, weights, bias = ctx.saved_tensors
        return mnist_cpp_model.linear_backward(input, weights, bias, grad_output)


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
        return MyLinearFunction.apply(input, self.weight, self.bias)


class MySoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(input: torch.Tensor):
        return mnist_cpp_model.softmax(input)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output = ctx.saved_tensors
        return mnist_cpp_model.d_softmax(output) * grad_output
    

class MySoftmax(torch.nn.Module):
        
        def forward(self, input: torch.Tensor):
            return MySoftmaxFunction.apply(input)


class MyReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(input: torch.Tensor):
        return mnist_cpp_model.relu(input)

    @staticmethod
    def setup_context(ctx, input, output):
        ctx.save_for_backward(input)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input = ctx.saved_tensors
        return mnist_cpp_model.d_relu(input) * grad_output
    

class MyReLU(torch.nn.Module):
    
        def forward(self, input: torch.Tensor):
            return MyReLUFunction.apply(input)