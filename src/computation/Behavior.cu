#include "computation/Behavior.h"
#include "computation/Kernels.h"
#include "Config.hpp"

#include <chrono>
#include <cuda_gl_interop.h>
#include <random>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace computation
{

	Behavior::Behavior(Config& config, GLuint shoalBuffer, FishProperties& properties) : _shoalBuffer(shoalBuffer), _propertiesHost(properties), _config(config)
	{
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

		// Allocate GPU buffers to general config
		cudaStatus = cudaMalloc(&_configDevice, sizeof(Config));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(_configDevice, &config, sizeof(Config), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
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
		cudaStatus = cudaMalloc(&_velocitiesDevice, _config.FISH_COUNT * 3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemset(_velocitiesDevice, 0, _config.FISH_COUNT * 3 * sizeof(float)); // regular fish start with speed 0

		// Generate speeds for predators
		float* predatorsVx = new float[_config.PREDATOR_COUNT];
		float* predatorsVy = new float[_config.PREDATOR_COUNT];
		float* predatorsVz = new float[_config.PREDATOR_COUNT];
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> distribution(_config.PREDATOR_MIN_SPEED, _config.PREDATOR_MAX_SPEED);
		for (int i = 0; i < _config.PREDATOR_COUNT; i++) {
			predatorsVx[i] = distribution(gen);
			predatorsVy[i] = distribution(gen);
			predatorsVz[i] = distribution(gen);
		}
		// Overwrite 0s for predators
		cudaStatus = cudaMemcpy(&_velocitiesDevice[_config.SHOAL_SIZE], predatorsVx, sizeof(float) * _config.PREDATOR_COUNT, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}
		cudaStatus = cudaMemcpy(&_velocitiesDevice[_config.FISH_COUNT + _config.SHOAL_SIZE], predatorsVy, sizeof(float) * _config.PREDATOR_COUNT, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}
		cudaStatus = cudaMemcpy(&_velocitiesDevice[_config.FISH_COUNT * 2 + _config.SHOAL_SIZE], predatorsVz, sizeof(float) * _config.PREDATOR_COUNT, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}
		delete[] predatorsVx;
		delete[] predatorsVy;
		delete[] predatorsVz;

		// Allocate GPU buffers for fish ids
		cudaStatus = cudaMalloc(&_fishIdsDevice, _config.SHOAL_SIZE * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}

		// Allocate GPU buffers for region indexes
		cudaStatus = cudaMalloc(&_regionIndexesDevice, _config.SHOAL_SIZE * sizeof(int) * 2);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}

		// Allocate GPU buffers for region starts
		cudaStatus = cudaMalloc(&_regionStartsDevice, _config.REGION_COUNT * sizeof(int));
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
	}

	Behavior::~Behavior()
	{
		cudaFree(_configDevice);
		cudaFree(_propertiesDevice);
		cudaFree(_velocitiesDevice);
		cudaFree(_fishIdsDevice);
		cudaFree(_regionIndexesDevice);
		cudaFree(_regionStartsDevice);
		cudaFree(_regionsCheatSheetDevice);

		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
		}
	}

	cudaError_t Behavior::ComputeRegionsCheatSheet()
	{
		cudaError_t cudaStatus;
		// Allocate GPU buffers for regions cheat sheet
		cudaStatus = cudaMalloc(&_regionsCheatSheetDevice, _config.REGION_COUNT * 27 * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return cudaStatus;
		}

		computeRegionsCheatSheetKernel << < _config.REGION_COUNT / _config.THREADS_PER_BLOCK + 1, _config.THREADS_PER_BLOCK >> > (_configDevice, _regionsCheatSheetDevice);

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
		assignFishToRegionsKernel << < _config.SHOAL_SIZE / _config.THREADS_PER_BLOCK + 1, _config.THREADS_PER_BLOCK >> > (_configDevice, (float*)positions_dev, _fishIdsDevice, _regionIndexesDevice);
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
		thrust::sort_by_key(thrust::device, _regionIndexesDevice, _regionIndexesDevice + _config.SHOAL_SIZE, _fishIdsDevice);
		findRegionStartsKernel << < _config.SHOAL_SIZE / _config.THREADS_PER_BLOCK + 1, _config.THREADS_PER_BLOCK >> > (_configDevice, _fishIdsDevice, _regionIndexesDevice, _regionStartsDevice);
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
		computeShoalMoveKernel << < _config.SHOAL_SIZE / _config.THREADS_PER_BLOCK + 1, _config.THREADS_PER_BLOCK >> > (_configDevice, (float*)positions_dev, _velocitiesDevice, _propertiesDevice, _fishIdsDevice, _regionIndexesDevice, _regionStartsDevice, _regionsCheatSheetDevice);
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
		computePredatorMoveKernel << < _config.PREDATOR_COUNT / _config.THREADS_PER_BLOCK + 1, _config.THREADS_PER_BLOCK >> > (_configDevice, (float*)positions_dev, _velocitiesDevice);
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
