#include "computation/Behavior.h"
#include "computation/Kernels.h"
#include <stdio.h>
#include <chrono>
#include <thread>
#include <cuda_gl_interop.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "Config.hpp"

namespace computation {

    Behavior::Behavior(GLuint shoalBuffer, FishProperties properties) : _shoalBuffer(shoalBuffer) { //TODO FREE
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

        // Allocate GPU buffers fo fish properties
        cudaStatus = cudaMalloc(&_propertiesDevice, sizeof(FishProperties));
        if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
        cudaStatus = cudaMemcpy(_propertiesDevice, &properties, sizeof(FishProperties), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return;
        }

        // Allocate GPU buffers for fish velocities
        cudaStatus = cudaMalloc(&_velocitiesDevice, sizeof(FishShoalVelocities));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return;
        }
        FishShoalVelocities velocities;
        std::fill(velocities.velocityX, velocities.velocityX + Config::SHOAL_SIZE, 0.001f);
        std::fill(velocities.velocityY, velocities.velocityY + Config::SHOAL_SIZE, 0.000f);
        std::fill(velocities.velocityZ, velocities.velocityZ + Config::SHOAL_SIZE, 0.000f);
        cudaStatus = cudaMemcpy(_velocitiesDevice, &velocities, sizeof(FishShoalVelocities), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return;
        }

        // Allocate GPU buffers for fish ids
        cudaStatus = cudaMalloc(&_fishIdsDevice, Config::SHOAL_SIZE * sizeof(int));
        if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}

        // Allocate GPU buffers for region indexes
		cudaStatus = cudaMalloc(&_regionIndexesDevice, Config::SHOAL_SIZE * sizeof(int) * 2);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return;
        }

        // Allocate GPU buffers for region starts
        cudaStatus = cudaMalloc(&_regionStartsDevice, Config::REGION_COUNT * sizeof(int));
        if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}

        cudaStatus = ComputeRegionsCheatSheet();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return;
        }
    }

    Behavior::~Behavior() {}

    cudaError_t Behavior::ComputeRegionsCheatSheet()
    {
        cudaError_t cudaStatus;
        // Allocate GPU buffers for regions cheat sheet
        cudaStatus = cudaMalloc(&_regionsCheatSheetDevice, Config::REGION_COUNT * 27 * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return cudaStatus;
        }

        int regionsCheatSheet[Config::REGION_COUNT * 27];
        int dim = Config::REGION_DIM_COUNT;
        for (int x = 0; x < dim; x++)
        {
            for (int y = 0; y < dim; y++)
            {
                for (int z = 0; z < dim; z++)
                {
                    int globalIndex = x * dim * dim + y * dim + z;
                    int regionsToCheckIndex = 0;

                    for (int i = -1; i <= 1; i++)
                    {
                        for (int j = -1; j <= 1; j++)
                        {
                            for (int k = -1; k <= 1; k++)
                            {
                                int xIndex = x + i;
                                int yIndex = y + j;
                                int zIndex = z + k;

                                if (xIndex >= 0 && xIndex < dim && yIndex >= 0 && yIndex < dim && zIndex >= 0 && zIndex < dim)
                                {
                                    regionsCheatSheet[globalIndex * 27 + regionsToCheckIndex] = xIndex * dim * dim + yIndex * dim + zIndex;
                                    regionsToCheckIndex++;
                                }
                            }
                        }
                    }

                    for (int i = regionsToCheckIndex; i < 27; i++)
					{
						regionsCheatSheet[globalIndex * 27 + i] = -1;
					}
                }
            }
        }
        
        cudaStatus = cudaMemcpy(_regionsCheatSheetDevice, regionsCheatSheet, Config::REGION_COUNT * 27 * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return cudaStatus;
		}

        return cudaSuccess;
    }

    cudaError_t Behavior::ComputeMove()
    {
        cudaError_t cudaStatus;

        cudaStatus = cudaGraphicsMapResources(1, &_resource, 0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsMapResources failed!");
            return cudaStatus;
        }

        void* positions_dev;
        size_t size;
        cudaStatus = cudaGraphicsResourceGetMappedPointer(&positions_dev, &size, _resource);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!");
            return cudaStatus;
        }

        // Assign fish to regions
        assignFishToRegionsKernel << <1, Config::SHOAL_SIZE >> > ((float*)positions_dev, _fishIdsDevice, _regionIndexesDevice);
        thrust::sort_by_key(thrust::device, _regionIndexesDevice, _regionIndexesDevice + Config::SHOAL_SIZE, _fishIdsDevice);
        findRegionStartsKernel << <1, Config::SHOAL_SIZE >> > (_fishIdsDevice, _regionIndexesDevice, _regionStartsDevice);

        // Compute movement
        computeMoveKernel << <1, Config::SHOAL_SIZE >> > ((float*)positions_dev, _velocitiesDevice, _propertiesDevice, _fishIdsDevice, _regionIndexesDevice, _regionStartsDevice, _regionsCheatSheetDevice);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            return cudaStatus;
        }

        float output4[Config::SHOAL_SIZE * 3];
        cudaStatus = cudaMemcpy(output4, positions_dev, size, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }

        // for debugging
        int output[Config::SHOAL_SIZE * 2];
        cudaStatus = cudaMemcpy(output, _regionIndexesDevice, Config::SHOAL_SIZE * sizeof(int) * 2, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }
        // for debugging
        int output2[Config::SHOAL_SIZE];
        cudaStatus = cudaMemcpy(output2, _fishIdsDevice, Config::SHOAL_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }
        // for debugging
        int output3[Config::REGION_COUNT];
        cudaStatus = cudaMemcpy(output3, _regionStartsDevice, Config::REGION_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }
        // for debugging
        int output5[Config::REGION_COUNT * 27];
        cudaStatus = cudaMemcpy(output5, _regionsCheatSheetDevice, Config::REGION_COUNT * 27 * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }

        FishShoalVelocities velocities;
        cudaStatus = cudaMemcpy(&velocities, _velocitiesDevice, sizeof(velocities), cudaMemcpyDeviceToHost);
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
