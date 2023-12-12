#include <pybind11/pybind11.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>

namespace py = pybind11;

__global__ void convolution2d_kernel(
    const float* input,
    const float* weights,
    float* output,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size
) {
    // Compute output indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width) {
        float sum = 0.0f;

        // Compute the convolution operation
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int input_row = row + i;
                int input_col = col + j;
                sum += input[input_row * input_width + input_col] *
                       weights[i * kernel_size + j];
            }
        }

        output[row * output_width + col] = sum;
    }
}

at::Tensor relu(at::Tensor x) {
    return at::relu(x);
}

at::Tensor d_relu(at::Tensor x) {
    return at::where(x > 0, at::ones_like(x), at::zeros_like(x));
}

at::Tensor softmax(at::Tensor x) {
    return at::softmax(x, 1);
}

at::Tensor d_softmax(at::Tensor x) {
    return at::softmax(x, 1) * (1 - at::softmax(x, 1));
}

at::Tensor linear_forward(at::Tensor x, at::Tensor w, at::Tensor b) {
    return at::addmm(b, x, w.t());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor d_out) {
    auto d_x = at::mm(d_out, w);
    auto d_w = at::mm(d_out.t(), x);
    auto d_b = d_out.sum(0);
    return std::make_tuple(d_x, d_w, d_b);
}

PYBIND11_MODULE(mnist_cpp_model, m) {
    m.def("relu", &relu, "ReLU activation function");
    m.def("d_relu", &d_relu, "ReLU derivative function");
    m.def("softmax", &softmax, "Softmax activation function");
    m.def("d_softmax", &d_softmax, "Softmax derivative function");
    m.def("linear_forward", &linear_forward, "Linear forward function");
    m.def("linear_backward", &linear_backward, "Linear backward function");
}