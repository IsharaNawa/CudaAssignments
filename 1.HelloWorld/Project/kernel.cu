#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda() {

    printf("Hello World\n");

}

int main() {

    // lanuch the kernal
    // since the host function does not have to wait for the kernal execution
    // this is called a asynchronous kernal launch
    // since only one printing statement , one thread is enough
    hello_cuda << <1, 1 >> > ();

    // wait for the kernal to finish its execution and then execute rest
    // for that , below line is used
    cudaDeviceSynchronize();

    //reset the device
    cudaDeviceReset();

    return 0;
}