#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "mnist_model.h"
#include <stdio.h>

namespace py = pybind11;

py::array_t<float> madd(py::array_t<float> x, py::array_t<float> y) {
    auto buf1 = x.request();
    auto buf2 = y.request();

    if (buf1.ndim != 2 || buf2.ndim != 2){
        printf("buf1 => %d\n", buf1.ndim);
        printf("buf2 => %d\n", buf2.ndim);
        throw std::runtime_error("Number of dimensions must be two");
    }

    if (buf1.size != buf2.size) {
        printf("buf1 => %d, %d\n", buf1.shape[0], buf1.shape[1]);
        printf("buf2 => %d, %d\n", buf2.shape[0], buf2.shape[1]);
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

py::array_t<float> mmelem(py::array_t<float> x, py::array_t<float> y) {
    auto buf1 = x.request();
    auto buf2 = y.request();

    if (buf1.ndim != 2 || buf2.ndim != 2) {
        printf("buf1 => %d\n", buf1.ndim);
        printf("buf2 => %d\n", buf2.ndim);
        throw std::runtime_error("Number of dimensions must be two");
    }

    if (buf1.size != buf2.size) {
        printf("buf1 => %d, %d\n", buf1.shape[0], buf1.shape[1]);
        printf("buf2 => %d, %d\n", buf2.shape[0], buf2.shape[1]);
        throw std::runtime_error("Input arrays must match");
    }

    int M = buf1.shape[1];
    int N = buf1.shape[0];
    auto out = py::array_t<float>(buf1.size);
    auto buf3 = out.request();

    float *A = (float *) buf1.ptr;
    float *B = (float *) buf2.ptr;
    float *C = (float *) buf3.ptr;

    cu_mmelem(A, B, C, M, N);

    return out;
}

py::array_t<float> mmreduce(py::array_t<float> x) {
    auto buf1 = x.request();

    if (buf1.ndim != 2) {
        printf("buf1 => %d\n", buf1.ndim);
        throw std::runtime_error("Number of dimensions must be two");
    }

    int M = buf1.shape[1];
    int N = buf1.shape[0];
    auto out = py::array_t<float>(M);
    auto buf2 = out.request();

    float *A = (float *) buf1.ptr;
    float *B = (float *) buf2.ptr;
    
    cu_mmreduce(A, B, M, N);

    return out;
}

py::array_t<float> mm(py::array_t<float> x, py::array_t<float> y) {
    auto buf1 = x.request();
    auto buf2 = y.request();

    if (buf1.ndim != 2 || buf2.ndim != 2) {
        printf("buf1 => %d\n", buf1.ndim);
        printf("buf2 => %d\n", buf2.ndim);
        throw std::runtime_error("Number of dimensions must be two");
    }

    if (buf1.shape[1] != buf2.shape[0]) {
        printf("buf1 => %d, %d\n", buf1.shape[0], buf1.shape[1]);
        printf("buf2 => %d, %d\n", buf2.shape[0], buf2.shape[1]);
        throw std::runtime_error("Matrices must have matching dimensions");
    }

    int N_a = buf1.shape[0];
    int M_a = buf1.shape[1];
    int M_b = buf2.shape[1];
    auto out = py::array_t<float>(N_a * M_b);
    auto buf3 = out.request();

    float *A = (float *) buf1.ptr;
    float *B = (float *) buf2.ptr;
    float *C = (float *) buf3.ptr;

    cu_mm(A, B, C, N_a, M_a, M_b);

    return out;
}

py::array_t<float> mt(py::array_t<float> x) {
    auto buf1 = x.request();

    if (buf1.ndim != 2) {
        printf("buf1 => %d\n", buf1.ndim);
        throw std::runtime_error("Number of dimensions must be two");
    }

    int N = buf1.shape[0];
    int M = buf1.shape[1];
    auto out = py::array_t<float>(buf1.size);
    auto buf2 = out.request();

    float *A = (float *) buf1.ptr;
    float *B = (float *) buf2.ptr;
    
    cu_mt(A, B, M, N);

    return out;
}

PYBIND11_MODULE(mnist_cpp_model, m) {
    m.def("madd", &madd, "Add two matrices");
    m.def("mmelem", &mmelem, "Multiply two matrices element-wise");
    m.def("mmreduce", &mmreduce, "Sum columns in a N x M matrix to produce a 1 x M matrix");
    m.def("mm", &mm, "Matrix multiplication");
    m.def("mt", &mt, "Transpose a matrix");
}