#pragma once

#include "glad/glad.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mutex>

namespace computation
{
    class Behavior
    {
    public:
        Behavior(GLuint shoalBuffer);
		~Behavior();

        cudaError_t ComputeMove();

    private:
        GLuint _shoalBuffer;
        cudaGraphicsResource* _resource;
    };
}
