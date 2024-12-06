#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "types.h"

#include <curand.h>
#include <curand_kernel.h>

namespace computation
{
	__global__ void computeRegionsCheatSheetKernel(Config* config, int* regionsCheatSheet);
	__global__ void assignFishToRegionsKernel(Config* config, float* positions, int* fishIds, int* regionIndexes);
	__global__ void findRegionStartsKernel(Config* config, int* fishIds, int* regionIndexes, int* regionStarts);
	__global__ void computeShoalMoveKernel(Config* config, float* positions, float* velocities, FishProperties* properties, int* fishIds, int* regionIndexes, int* regionStarts, int* regionsCheatSheet);
	__global__ void computePredatorMoveKernel(Config* config, float* positions, float* velocities);
}