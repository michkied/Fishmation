#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "types.h"

#include <curand.h>
#include <curand_kernel.h>

namespace computation {
	__global__ void computeRegionsCheatSheetKernel(int* regionsCheatSheet);
	__global__ void setupPredatorRandomnessKernel(curandState* state, PredatorVelocities* velocities);
	__global__ void assignFishToRegionsKernel(float* positions, int* fishIds, int* regionIndexes);
	__global__ void findRegionStartsKernel(int* fishIds, int* regionIndexes, int* regionStarts);
	__global__ void computeShoalMoveKernel(float* positions, FishShoalVelocities* velocities, FishProperties* properties, int* fishIds, int* regionIndexes, int* regionStarts, int* regionsCheatSheet);
	__global__ void computePredatorMoveKernel(float* positions, PredatorVelocities* velocities);
}