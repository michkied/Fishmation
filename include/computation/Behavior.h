#pragma once

#include "glad/glad.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "types.h"

#include <curand.h>
#include <curand_kernel.h>

namespace computation
{
    class Behavior
    {
    public:
        Behavior(GLuint shoalBuffer, FishProperties properties);
		~Behavior();

        cudaError_t ComputeMove();

    private:
        cudaError_t ComputeRegionsCheatSheet();
        cudaError_t SetupPredators();

        GLuint _shoalBuffer;
        cudaGraphicsResource* _resource;

        FishProperties* _propertiesDevice;
        FishShoalVelocities* _velocitiesDevice;

        int* _fishIdsDevice;
        int* _regionIndexesDevice;
        int* _regionStartsDevice;
        int* _regionsCheatSheetDevice;

        curandState* _predatorStateDevice;
        PredatorVelocities* _predatorVelocitiesDevice;
    };
}
