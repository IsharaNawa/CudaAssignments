#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

/* NOTE : THIS CODE IS NOT COMPLETE. 3D INDEX IS NOT FOUND CORRECTLY*/

// threads and elements are equal hence size is not needed to the kernal
__global__ void mem_transfer_test(int* input) {

    int one_layer_of_block = blockDim.x * blockDim.y;
    int tid = threadIdx.x + blockDim.y * threadIdx.y + threadIdx.z * one_layer_of_block;

    int block_size = blockDim.x * blockDim.y * blockDim.z;

    //printf("threadIdx.x %d threadIdx.y : %d threadIdx.z : %d tid : %d\n ", threadIdx.x, threadIdx.y, threadIdx.z,tid);

    int one_layer_of_grid = block_size * gridDim.x * gridDim.y;

    // one dimentional block with two blocks
    int gid = tid + blockIdx.x + gridDim.y * blockIdx.y + blockIdx.z * one_layer_of_grid;

    // first check if the index is less than the size
    printf("tid : %d , gid : %d , value : %d \n", tid, gid, input[gid]);
    
}

int main()
{
    // size of the array
    int size = 64;

    // get the byte size
    int byte_size = size * sizeof(int);

    // array in the host
    int* h_input;

    //allocate memory for host array
    h_input = (int*)malloc(byte_size);

    // randomly populate the array

    // set the seed
    time_t t;
    srand((unsigned)time(&t));

    // populate the array
    for (int i = 0;i < size;i++) {
        h_input[i] = (int)(rand() & 0xff);
    }

    // allocate memory for the device
    int* d_input;

    // allocate memory in the device
    cudaMalloc((void**)&d_input, byte_size);

    // copy the array to the device from the host
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // block is a one dimentional 64 threads
    dim3 block(2,2,2);

    // there are only 5 blocks in the grid
    dim3 grid(2,2,2);

    // lanuch the kernal
    mem_transfer_test << < grid, block >> > (d_input);

    // wait until kernal execution finished
    cudaDeviceSynchronize();

    // reset the device
    cudaDeviceReset();

    // free allocated memory in the device
    cudaFree(d_input);

    // free allocated memory in the host
    free(h_input);

    return 0;
}


