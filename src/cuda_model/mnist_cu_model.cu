#include <cuda_runtime.h>
#include <stdio.h>
#include "mnist_model.h"

__global__ void kernel_madd(float* A, float* B, float* C, int M, int N);
__global__ void kernel_mmelem(float* A, float* B, float* C, int M, int N);

void cu_madd(float* A, float* B, float* C, int M, int N)
{
	float *d_a, *d_b, *d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;

	int size = sizeof(float)*M*N;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

	kernel_madd << < grid, blk >> > (d_a, d_b, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void cu_mmelem(float* A, float* B, float* C, int M, int N)
{
	float *d_a, *d_b, *d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;

	int size = sizeof(float)*M*N;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

	kernel_madd << < grid, blk >> > (d_a, d_b, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void kernel_madd(float* A, float* B, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;

	if (ix < M && iy < N)
		C[idx] = A[idx] + B[idx];
}

__global__ void kernel_mmelem(float* A, float* B, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;

	if (ix < M && iy < N)
		C[idx] = A[idx] * B[idx];
}