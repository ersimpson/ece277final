# ECE 277 Final Project Fall 2023

## Eric Simpson, A59020270

### Overview

For my final project, I am implementing a simple 2 layer fully connected neural network that will classify the MNIST digits dataset. The hidden layer has 784 input features and 128 output features,
and the output layer has 128 input features and 10 output features. The activation function used in
the hidden layer is a sigmoid function and the activation function for the output layer is a softmax function.

While this architecture clearly is not state of the art, it is intentionally kept simple to highlight the integration of custom CUDA kernels as C++ extensions to a Python frontend. In particular, I use Numpy as the Python frontend framework for developing the network and custom CUDA kernels are written to replace many of the common matrix operations abundant in neural networks (e.g. matrix addition, matrix transposes, etc.).

### Prerequisites

- CUDA Enabled Machine
- PyBind11
- Windows 10 x64
- CMake
- Visual Studio 16 2019
- Python3.9

### Project Setup

a. Install required python packages. NOTE: CUDA enabled PyTorch not needed as PyTorch is only used for loading datasets.

`python3 -m pip install --user torch torchaudio torchvision numpy`

b. Build project with CMake.

/path/to/ece277final -> Configure -> Generate -> Open Project

c. Build Solution in Visual Studio

Release -> Build -> Build Solution

### Running Project

There is one primary script in which a model for the network can be trained. Note there is currently not a feature to save off the weights of a model as the purposes of this project is to showcase the custom CUDA kernels integration with Python and the training is simply for demonstration purposes.

It is recommended to train the model with only batch sizes of <10000, otherwise out of memory errors may occur when operating with the entire training dataset.

To train a model entirely on the CPU with just Numpy:

`python3 -m py_src.cli --epochs=100 --batch-size=10000`

To train a model with the matrix operations implemented with custom CUDA kernels:

`python3 -m py_src.cli --epochs=100 --batch-size=10000 --case=2`

### Results

After training 100 epochs, the training precision is around 0.84 and shows signs of converging further. Training the model for 100 epochs takes around ~10 minutes.

Note that the training times with the custom CUDA kernels is very similar to that of the default case; however, this is likely due to the time wasted copying data from the host to device and back.

### Final Considerations

While this code showcases the integration of custom CUDA kernels with PyBind11 and Numpy for nerual networks, there is still plenty of room to improve the optimization of the custom CUDA kernels. In particular, we could have written the entire network with CUDA kernels storing the weights on the device and only copying back to the host when the training was complete. Note I avoided this approach as I originally wanted to showcase integrating custom CUDA kernels with a PyTorch frontend and take advantage of the niceties of the PyTorch framework such as the auto-differentiation.

That said, if I was afforded more time there are some additional optimizations we can make to the custom CUDA kernels. In no order of importance:
    - Use concurrent streams and pinned memory for copy operations and kernels
    - Use warp execution for stencil functions to apply vector reduce operations
    - Use shared memory for matrix multiplication

