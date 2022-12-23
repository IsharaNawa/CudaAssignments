#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_threadIds(){

    // print the threadIdx , blockIdx and gridDim variables
    printf("threadIdx.x: %d threadIdx.y: %d threadIdx.z: %d blockIdx.x: %d blockIdx.y: %d blockIdx.z: %d gridDim.x: %d gridDim.y: %d gridDim.z: %d",
    threadIdx.x,threadIdx.y,threadIdx.z,
    blockIdx.x,blockIdx.y,blockIdx.z,
    gridDim.x,gridDim.y,gridDim.z);

}

int main(){

    // define the number of threads in the grid x
    int numberOfThreadsAlongX = 4;

    // define the number of threads in the grid y
    int numberOfThreadsAlongY = 4;

    // define the number of threads in the grid z
    int numberOfThreadsAlongZ = 4;

    // now define the block size
    dim3 block(2,2,2);

    // now define the gird
    dim3 grid(numberOfThreadsAlongX/block.x,numberOfThreadsAlongY/block.y,
                numberOfThreadsAlongZ/block.z);

    // now call the kernal
    print_threadIds <<< grid,block >>> ();

    // now synchroize the code
    cudaDeviceSynchronize();

    // reset the device
    cudaDeviceReset();

    return 0;
}