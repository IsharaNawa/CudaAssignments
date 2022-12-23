#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_value_using_unique_threadId(int* input) {

    // get the threadId
    int tid = threadIdx.x;

    // print the value
    printf("threadIdx: %d , value : %d\n", tid, input[tid]);

}

int main() {


    // define array size 
    int size = 8;

    // calculate the number of bytes for the array
    int array_byte_size = sizeof(int) * size;

    // get a new array
    int h_data[] = { 23,9,4,55,65,12,1,33 };

    // print digits
    for (int i = 0;i < size;i++) {
        printf("%d ", h_data[i]);
    }

    printf("\n");

    // define an array for data in device
    int * d_data;

    // get space for data inside the device
    cudaMalloc((void**)&d_data, array_byte_size);

    // copy the values from host to device
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    // now define the block size
    dim3 block(8);

    // now define the gird
    dim3 grid(1);

    // now call the kernal
    print_value_using_unique_threadId << < grid, block >> > (d_data);

    // now synchroize the code
    cudaDeviceSynchronize();

    // reset the device
    cudaDeviceReset();

    return 0;
}