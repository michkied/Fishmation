#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "types.h"

namespace computation {
	__global__ void assignFishToRegionsKernel(float* positions, int* fishIds, int* regionIndexes);
	__global__ void findRegionStartsKernel(int* fishIds, int* regionIndexes, int* regionStarts);
	__global__ void computeMoveKernel(float* positions, FishShoalVelocities* velocities, FishProperties* properties, int* fishIds, int* regionIndexes, int* regionStarts, int* regionsCheatSheet);
}