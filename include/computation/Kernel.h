#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace computation {
	__global__ void computeMoveKernel(float* positions);
}