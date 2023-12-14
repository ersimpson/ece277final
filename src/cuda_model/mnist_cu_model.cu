#include <cuda_runtime.h>
#include <stdio.h>
#include "mnist_model.h"

__global__ void kernel_madd(float* A, float* B, float* C, int M, int N);
__global__ void kernel_mmelem(float* A, float* B, float* C, int M, int N);
__global__ void kernel_mmreduce(float* A, float* B, int M, int N);
__global__ void kernel_mm(float* A, float* B, float* C, int N_a, int M_a, int M_b);
__global__ void kernel_mt(float* A, float* B, int M, int N);


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

void cu_mm(float* A, float* B, float* C, int N_a, int M_a, int M_b)
{
	float *d_a, *d_b, *d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M_b + blk.x - 1) / blk.x;
	grid.y = (N_a + blk.y - 1) / blk.y;

	int sizeA = sizeof(float)*M_a*N_a;
    int sizeB = sizeof(float)*M_b*M_a;
	int sizeC = sizeof(float)*N_a*M_b;

	cudaMalloc((void **)&d_a, sizeA);
	cudaMalloc((void **)&d_b, sizeB);
	cudaMalloc((void **)&d_c, sizeC);

	cudaMemcpy(d_a, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, sizeB, cudaMemcpyHostToDevice);

	kernel_mm << < grid, blk >> > (d_a, d_b, d_c, N_a, M_a, M_b);

	cudaMemcpy(C, d_c, sizeC, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void cu_mt(float* A, float* B, int M, int N)
{
	float *d_a, *d_b;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;

	int size = sizeof(float)*M*N;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	
	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);

	kernel_mt << < grid, blk >> > (d_a, d_b, M, N);

	cudaMemcpy(B, d_b, size, cudaMemcpyDeviceToHost);

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

	float sum = 0.0f;
	for (int i = 0; i < N; i++) {
        sum += A[i * M + ix];
	}
	B[ix] = sum;
}

__global__ void kernel_mm(float *A, float *B, float *C, int N_a, int M_a, int M_b) 
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (iy >= N_a || ix >= M_b)
		return;

	float sum = 0.0f;
	for (int i = 0; i < M_a; i++)
		sum += A[iy * M_a + i] * B[i * M_b + ix];
	C[iy * M_b + ix] = sum;
}

__global__ void kernel_mt(float* A, float* B, int M, int N)
{
	__shared__ float tile[16][16];

	// Coordinates of original matrix
	unsigned int ix, iy, ti, to;
	ix = threadIdx.x + blockIdx.x * blockDim.x;
	iy = threadIdx.y + blockIdx.y * blockDim.y;
	ti = iy * M + ix;
	
	if (iy < N && ix < M)
	{
		tile[threadIdx.y][threadIdx.x] = A[ti]; // load to shared memory
	}
	
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;
	// Coordinates of transposed matrix
	ix = blockIdx.y * blockDim.y + icol;
	iy = blockIdx.x * blockDim.x + irow;
	to = iy * N + ix;
	__syncthreads();

	if (iy < M && ix < N) {
		B[to] = tile[icol][irow]; // load back to global memory
	}
}