#include <pybind11/pybind11.h>
#include <ATen/ATen.h>

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

at::Tensor avg_pool(at::Tensor x) {
    return at::avg_pool2d(x, 2);
}

at::Tensor d_avg_pool(at::Tensor x) {
    return at::upsample_bilinear2d(x, at::IntList{2, 2});
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

at::Tensor conv_forward(at::Tensor x, at::Tensor w, at::Tensor b) {
    return at::conv2d(x, w, b, at::IntList{1, 1});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_backward(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor d_out) {
    auto d_x = at::conv_transpose2d(d_out, w, b, at::IntList{1, 1});
    auto d_w = at::conv2d(x, d_out, at::IntList{1, 1});
    auto d_b = d_out.sum(at::IntList{0, 2, 3});
    return std::make_tuple(d_x, d_w, d_b);
}

PYBIND11_MODULE(mnist_cpp_model, m) {
    m.def("relu", &relu, "ReLU activation function");
    m.def("d_relu", &d_relu, "ReLU derivative function");
    m.def("softmax", &softmax, "Softmax activation function");
    m.def("d_softmax", &d_softmax, "Softmax derivative function");
    m.def("avg_pool", &avg_pool, "Average pooling function");
    m.def("d_avg_pool", &d_avg_pool, "Average pooling derivative function");
    m.def("linear_forward", &linear_forward, "Linear forward function");
    m.def("linear_backward", &linear_backward, "Linear backward function");
    m.def("conv_forward", &conv_forward, "Convolution forward function");
    m.def("conv_backward", &conv_backward, "Convolution backward function");
}