#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_threadIds(){

    printf("threadIdx.x: %d threadIdx.y: %d threadIdx.z: %d",threadIdx.x,threadIdx.y,threadIdx.z);

}

int main(){

    // define the number of threads in the grid x
    int numberOfThreadsAlongX = 16;

    // define the number of threads in the grid y
    int numberOfThreadsAlongY = 16;

    // now define the block size
    dim3 block(8,8);

    // now define the gird
    dim3 grid(numberOfThreadsAlongX/block.x,numberOfThreadsAlongY/block.y);

    // now call the kernal
    print_threadIds <<< grid,block >>> ();

    // wait for the kernal to finish
    cudaDeviceSynchronize();

    // reset the gpu
    cudaDeviceReset();

    return 0;
}