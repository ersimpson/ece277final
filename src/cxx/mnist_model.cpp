#include <torch/extension.h>

#include <iostream>
#include <cmath>


torch::Tensor d_relu(torch::Tensor x) {
    auto ptr = tensor.data_ptr<float>();
    int size = tensor.numel();

    for (int i = 0; i < size; i++) {
        ptr[i] = (ptr[i] > 0) ? ptr[i] : 0;
    }
    return x;
}

torch::Tensor d_softmax(torch::Tensor x) {
    auto ptr = tensor.data_ptr<float>();
    int size = tensor.numel();

    auto sum = 0;
    for (int i = 0; i < size; i++) {
        ptr[i] = std::exp(ptr[i]);
        sum += ptr[i];
    }
    for (int i = 0; i < size; i++) {
        ptr[i] = ptr[i] / sum;
    }

    return x;
}

torch::Tensor d_linear_forward(torch::Tensor x) {
    auto ptr = tensor.data_ptr<float>();
    int size = tensor.numel();

    for (int i = 0; i < size; i++) {
        ptr[i] = ptr[i];
    }
    return x;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", &d_relu, "ReLU activation function");
    m.def("softmax", &d_softmax, "Softmax activation function");
}