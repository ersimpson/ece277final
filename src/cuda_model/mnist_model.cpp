#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "mnist_model.h"

namespace py = pybind11;

py::array_t<double> madd(py::array_t<double> x, py::array_t<double> y) {
    auto buf1 = x.request();
    auto buf2 = y.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input arrays must match");

    int M = buf1.shape[0];
    int N = buf1.shape[1];
    auto out = py::array_t<double>(buf1.size);
    auto buf3 = out.request();

    double *A = (double *) buf1.ptr;
    double *B = (double *) buf2.ptr;
    double *C = (double *) buf3.ptr;

    cu_madd(A, B, C, M, N);

    return out;
}


PYBIND11_MODULE(mnist_cpp_model, m) {
    m.def("madd", &madd, "Add two matrices");
}