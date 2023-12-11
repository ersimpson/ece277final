#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

py::array_t<double> softmax(py::array_t<double> x) {
	int N = x.shape()[0];
	int M = x.shape()[1];
	printf("N=%d, M=%d\n", N, M);

	auto result = py::array(py::buffer_info(
		nullptr,
		sizeof(double),
		py::format_descriptor<double>::value,
		2,
		{ N, 1 },
		{ sizeof(double) * N,  sizeof(double) }
	));

	auto x_buf = x.request();
	auto res_buf = result.request();

	double sum = 0;
	double* x_ptr = (double*)x_buf.ptr;
	double* res_ptr = (double*)res_buf.ptr;
	for (int i = 0; i < N; i++) {
		res_ptr[i] = std::exp(x_ptr[i]);
		sum += res_ptr[i];
	}
    for (int i = 0; i < N; i++) {
        res_ptr[i] = res_ptr[i] / sum;
    }

    return result;
}

PYBIND11_MODULE(mnist_cpp_model, m) {
    m.def("softmax", &softmax, "Softmax activation function");
}