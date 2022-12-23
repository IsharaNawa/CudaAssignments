#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void mem_transfer_test(int* input,int size) {

    // one dimentional block with two blocks
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // first check if the index is less than the size
    if (gid < size) {
        // print the value
        printf("tid : %d , gid : %d , value : %d \n", threadIdx.x, gid, input[gid]);
    }
}

int main()
{
    // size of the array
    int size = 150;

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
    dim3 block(32);

    // there are only 5 blocks in the grid
    dim3 grid(5);

    // lanuch the kernal
    mem_transfer_test << < grid, block >> > (d_input,size);

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


