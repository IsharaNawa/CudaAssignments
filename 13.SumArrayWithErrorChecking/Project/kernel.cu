#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "cuda_common.cuh"

#include <stdlib.h>
#include <time.h>
#include <cstring>

__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {

	// since grids and blocks are 1D
	int gid = threadIdx.x + blockDim.x * blockIdx.x;

	// do the calculation
	if (gid < size) {
		c[gid] = a[gid] + b[gid];
	}

}

// for validity checking
void sum_array_cpu(int* a, int* b, int* c, int size) {

	for (int i = 0;i < size;i++) {
		c[i] = b[i] + a[i];
	}

}

// for validity checking
void compare_arrays(int* a, int* b, int size) {

	for (int i = 0;i < size; i++) {

		if (a[i] != b[i]) {
			printf("Arrays are different i:%d %d %d\n", i, a[i], b[i]);
			return;
		}

	}

	printf("Arrays are the same \n");
}

int main() {

	// size of the array
	int size = 10000;

	// calculate the byte size
	int bytes_size = size * sizeof(int);

	// get the block size
	int block_size = 128;

	// host array pointers and allocation
	int* h_a = (int*)malloc(bytes_size);
	int* h_b = (int*)malloc(bytes_size);

	// to store gpu calculations
	int* gpu_results = (int*)malloc(bytes_size);

	// randomly initialize pointers
	time_t t;
	srand((unsigned)time(&t));

	// assign values to arrays
	for (int i = 0;i < size;i++) {
		h_a[i] = (int)(rand() & 0xff);
		//h_b[i] = (int)(rand() & 0xff);
	}

	for (int i = 0;i < size;i++) {
		//h_a[i] = (int)(rand() & 0xff);
		h_b[i] = (int)(rand() & 0xff);
	}

	// set the gpu results to 0 initially
	memset(gpu_results, 0, bytes_size);

	// device pointers
	int* d_a, * d_b, * d_c;

	// allocate memory on the device
	gpuErrchk(cudaMalloc((int**)&d_a, bytes_size));
	gpuErrchk(cudaMalloc((int**)&d_b, bytes_size));
	gpuErrchk(cudaMalloc((int**)&d_c, bytes_size));

	// copy the h_a and h_b pointers to the device
	gpuErrchk(cudaMemcpy(d_a, h_a, bytes_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, bytes_size, cudaMemcpyHostToDevice));

	// launching the grid
	dim3 block(block_size);

	// we should add one to make sure that there are more blocks than elements
	dim3 grid(size / block.x + 1);

	// launching the kernal
	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);

	// wait until the kernal execution is finished
	gpuErrchk(cudaDeviceSynchronize());

	// copy the result back to the host
	gpuErrchk(cudaMemcpy(gpu_results, d_c, bytes_size, cudaMemcpyDeviceToHost));

	// we dont have a way to confirm if the gpu implementation is correct or not
	// because the array is very large therefore can not print and check each value
	// therefore we need to check the gpu result with the cpu result
	// this is validity checking

	// create a new array for this
	int* h_c = (int*)malloc(bytes_size);

	// sum up using the cpu
	sum_array_cpu(h_a, h_b, h_c, size);

	// now we need to compare two arrays
	// but since that function will come in many places
	// it is ideal to put that function is a header file
	// calling the function from the header file
	compare_arrays(gpu_results, h_c, size);

	// reclaim the memory
	gpuErrchk(cudaFree(d_c));
	gpuErrchk(cudaFree(d_a));
	gpuErrchk(cudaFree(d_b));

	// reclainming the hsot memory
	free(h_a);
	free(h_b);
	free(gpu_results);

	// reset the device
	gpuErrchk(cudaDeviceReset());
	return 0;
}



