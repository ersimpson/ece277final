#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "mnist_model.h"
#include <stdio.h>

namespace py = pybind11;

py::array_t<float> madd(py::array_t<float> x, py::array_t<float> y) {
    auto buf1 = x.request();
    auto buf2 = y.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    if (buf1.size != buf2.size) {
        printf("buf1 => %d, %d\n", buf1.shape[0], buf1.shape[1]);
        printf("buf2 => %d, %d", buf2.shape[0], buf2.shape[1]);
        throw std::runtime_error("Input arrays must match");
    }

    int M = buf1.shape[1];
    int N = buf1.shape[0];
    auto out = py::array_t<float>(buf1.size);
    auto buf3 = out.request();

    float *A = (float *) buf1.ptr;
    float *B = (float *) buf2.ptr;
    float *C = (float *) buf3.ptr;

    cu_madd(A, B, C, M, N);

    return out;
}


PYBIND11_MODULE(mnist_cpp_model, m) {
    m.def("madd", &madd, "Add two matrices");
}