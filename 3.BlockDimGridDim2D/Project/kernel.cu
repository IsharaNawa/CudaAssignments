#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_threadIds() {

    printf("blockIdx.x: %d blockIdx.y: %d blockIdx.z: %d blockDim.x: %d blockDim.y: %d blockDim.z: %d gridDim.x: %d gridDim.y: %d gridDim.z: %d\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);

}

int main() {

    // define the number of threads in the grid x
    int numberOfThreadsAlongX = 16;

    // define the number of threads in the grid y
    int numberOfThreadsAlongY = 16;

    // now define the block size
    dim3 block(8, 8);

    // now define the gird
    dim3 grid(numberOfThreadsAlongX / block.x, numberOfThreadsAlongY / block.y);

    // now call the kernal
    print_threadIds << < grid, block >> > ();

    // now synchroize the code
    cudaDeviceSynchronize();

    // reset the device
    cudaDeviceReset();

    return 0;
}