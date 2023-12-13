#include <cuda_runtime.h>
#include <stdio.h>
#include "mnist_model.h"

__global__ void kernel_madd(float* A, float* B, float* C, int M, int N);
__global__ void kernel_mmelem(float* A, float* B, float* C, int M, int N);
__global__ void kernel_mmreduce(float* A, float* B, int M, int N);

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

	kernel_mmelem << < grid, blk >> > (d_a, d_b, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void cu_mmreduce(float* A, float* B, int M, int N)
{
	float *d_a, *d_b;

	int blk = 256;
    int grid = (M + blk - 1) / blk;
	int sizeA = sizeof(float)*M*N;
    int sizeB = sizeof(float)*M;

	cudaMalloc((void **)&d_a, sizeA);
	cudaMalloc((void **)&d_b, sizeB);

	cudaMemcpy(d_a, A, sizeA, cudaMemcpyHostToDevice);

	kernel_mmreduce << < grid, blk >> > (d_a, d_b, M, N);

	cudaMemcpy(A, d_b, sizeB, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
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

__global__ void kernel_mmreduce(float* A, float* B, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

	float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += A[i * M + ix];
    }
	B[ix] = sum;
}