#include "computation/Behavior.h"
#include "computation/Kernel.h"
#include <stdio.h>
#include <chrono>
#include <thread>
#include <cuda_gl_interop.h>
#include "Config.hpp"

namespace computation {

    Behavior::Behavior(GLuint shoalBuffer) : _shoalBuffer(shoalBuffer) {
        cudaError_t cudaStatus;

        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            return;
        }

        cudaStatus = cudaGraphicsGLRegisterBuffer(&_resource, _shoalBuffer, cudaGraphicsRegisterFlagsNone);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed!");
            return;
        }
    }

    Behavior::~Behavior() {}

    cudaError_t Behavior::ComputeMove()
    {
        cudaError_t cudaStatus;

        cudaStatus = cudaGraphicsMapResources(1, &_resource, 0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsMapResources failed!");
            return cudaStatus;
        }

        void* dev_ptr;
        size_t size;
        cudaStatus = cudaGraphicsResourceGetMappedPointer(&dev_ptr, &size, _resource);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!");
            return cudaStatus;
        }

        // Launch a kernel on the GPU with one thread for each element.
        computeMoveKernel << <1, Config::SHOAL_SIZE >> > ((float*)dev_ptr);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            return cudaStatus;
        }

        float output[Config::SHOAL_SIZE * 3];
        cudaStatus = cudaMemcpy(output, dev_ptr, size, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }

        cudaStatus = cudaGraphicsUnmapResources(1, &_resource, 0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsUnmapResources failed!");
            return cudaStatus;
        }

        return cudaStatus;
    }
}
