#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "glad/glad.h"
#include "types.h"

#include <curand.h>
#include <curand_kernel.h>

namespace computation
{
	class Behavior
	{
	public:
		Behavior(Config& config, GLuint shoalBuffer, FishProperties& properties);
		~Behavior();

		cudaError_t ComputeMove();

	private:
		cudaError_t ComputeRegionsCheatSheet();

		Config& _config;
		Config* _configDevice;

		FishProperties& _propertiesHost;
		FishProperties* _propertiesDevice;
		int _propertiesChangeCounter = _propertiesHost.changeCounter;

		GLuint _shoalBuffer;
		cudaGraphicsResource* _resource;

		float* _velocitiesDevice;

		int* _fishIdsDevice;
		int* _regionIndexesDevice;
		int* _regionStartsDevice;
		int* _regionsCheatSheetDevice;
	};
}
