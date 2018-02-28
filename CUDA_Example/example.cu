#include <iostream>

// Kernels ==============================================================

__global__ void kernel(int *a, int *b, unsigned int N) {
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<N) {
		a[i] += b[i] + 100;
		printf("GPU (thread %d): %d\n", i , a[i]);
	}
};

// Lanch kernels =======================================================

extern "C" void launch_kernel(int *a, int *b, unsigned int N){
	dim3 grid(1);
	dim3 block(N);
	kernel << <grid, block >> >(a, b, N);
};