#include "Config.hpp"
#include "computation/Behavior.h"
#include "computation/Kernels.h"

#include <stdio.h>
#include <chrono>
#include <random>
#include <cuda_gl_interop.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace computation {

    Behavior::Behavior(GLuint shoalBuffer, FishProperties& properties) : _shoalBuffer(shoalBuffer), _propertiesHost(properties) {
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
        std::fill(velocities.velocityX, velocities.velocityX + Config::SHOAL_SIZE, 0.0f);
        std::fill(velocities.velocityY, velocities.velocityY + Config::SHOAL_SIZE, 0.0f);
        std::fill(velocities.velocityZ, velocities.velocityZ + Config::SHOAL_SIZE, 0.0f);
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

        // Precompute lookup table for neighboring regions
        cudaStatus = ComputeRegionsCheatSheet();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "ComputeRegionsCheatSheet failed!");
            return;
        }

        // Allocate GPU buffers for predator velocities
        cudaStatus = cudaMalloc(&_predatorVelocitiesDevice, sizeof(PredatorVelocities));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distribution(Config::PREDATOR_MIN_SPEED, Config::PREDATOR_MAX_SPEED);
        PredatorVelocities predatorVelocities;
        for (int i = 0; i < Config::PREDATOR_COUNT; i++) {
			predatorVelocities.velocityX[i] = distribution(gen);
			predatorVelocities.velocityY[i] = distribution(gen);
			predatorVelocities.velocityZ[i] = distribution(gen);
		}
        cudaStatus = cudaMemcpy(_predatorVelocitiesDevice, &predatorVelocities, sizeof(PredatorVelocities), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}
    }

    Behavior::~Behavior() {
        cudaFree(_propertiesDevice);
		cudaFree(_velocitiesDevice);
		cudaFree(_fishIdsDevice);
		cudaFree(_regionIndexesDevice);
		cudaFree(_regionStartsDevice);
		cudaFree(_regionsCheatSheetDevice);
        cudaFree(_predatorVelocitiesDevice);

        cudaError_t cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
        }
    }

    cudaError_t Behavior::ComputeRegionsCheatSheet()
    {
        cudaError_t cudaStatus;
        // Allocate GPU buffers for regions cheat sheet
        cudaStatus = cudaMalloc(&_regionsCheatSheetDevice, Config::REGION_COUNT * 27 * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return cudaStatus;
        }

        computeRegionsCheatSheetKernel<<< Config::REGION_COUNT / Config::THREADS_PER_BLOCK + 1, Config::THREADS_PER_BLOCK >>>(_regionsCheatSheetDevice);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "computeRegionsCheatSheetKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeRegionsCheatSheetKernel!\n", cudaStatus);
            return cudaStatus;
        }

        return cudaSuccess;
    }

    cudaError_t Behavior::ComputeMove()
    {
        cudaError_t cudaStatus;

        if (_propertiesHost.changeCounter != _propertiesChangeCounter) {
            cudaStatus = cudaMemcpy(_propertiesDevice, &_propertiesHost, sizeof(FishProperties), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                return cudaStatus;
            }
            _propertiesChangeCounter = _propertiesHost.changeCounter;
        }

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
        assignFishToRegionsKernel<<< Config::SHOAL_SIZE / Config::THREADS_PER_BLOCK + 1, Config::THREADS_PER_BLOCK >>>((float*)positions_dev, _fishIdsDevice, _regionIndexesDevice);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "assignFishToRegionsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching assignFishToRegionsKernel!\n", cudaStatus);
            return cudaStatus;
        }
        thrust::sort_by_key(thrust::device, _regionIndexesDevice, _regionIndexesDevice + Config::SHOAL_SIZE, _fishIdsDevice);
        findRegionStartsKernel<<< Config::SHOAL_SIZE / Config::THREADS_PER_BLOCK + 1, Config::THREADS_PER_BLOCK >>>(_fishIdsDevice, _regionIndexesDevice, _regionStartsDevice);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "findRegionStartsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching findRegionStartsKernel!\n", cudaStatus);
            return cudaStatus;
        }

        // Compute shoal movement
        computeShoalMoveKernel<<< Config::SHOAL_SIZE / Config::THREADS_PER_BLOCK + 1, Config::THREADS_PER_BLOCK >>>((float*)positions_dev, _velocitiesDevice, _propertiesDevice, _fishIdsDevice, _regionIndexesDevice, _regionStartsDevice, _regionsCheatSheetDevice);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "computeShoalMoveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeShoalMoveKernel!\n", cudaStatus);
            return cudaStatus;
        }

        // Compute predator movement
        computePredatorMoveKernel << < Config::PREDATOR_COUNT / Config::THREADS_PER_BLOCK + 1, Config::THREADS_PER_BLOCK >> >((float*)positions_dev, _predatorVelocitiesDevice);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "computePredatorMoveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computePredatorMoveKernel!\n", cudaStatus);
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
