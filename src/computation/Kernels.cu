#include "computation/Kernels.h"
#include <cmath>

namespace computation {
    __global__ void assignFishToRegionsKernel(float* positions, int* fishIds, int* regionIndexes)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        int xIndex = i;
        int yIndex = i + Config::SHOAL_SIZE;
        int zIndex = i + Config::SHOAL_SIZE * 2;

        float tempX = positions[xIndex] / Config::REGION_SIZE;
        float tempY = positions[yIndex] / Config::REGION_SIZE;
        float tempZ = positions[zIndex] / Config::REGION_SIZE;

        int idX = (tempX >= 0) ? (int)tempX + 1 : (int)tempX - 1;
        int idY = (tempY >= 0) ? (int)tempY + 1 : (int)tempY - 1;
        int idZ = (tempZ >= 0) ? (int)tempZ + 1 : (int)tempZ - 1;

        int linearX = Config::REGION_DIM_COUNT / 2 - idX - (int)(idX < 0);
        int linearY = Config::REGION_DIM_COUNT / 2 - idY - (int)(idY < 0);
        int linearZ = Config::REGION_DIM_COUNT / 2 - idZ - (int)(idZ < 0);

        regionIndexes[i] = linearX + linearY * Config::REGION_DIM_COUNT + linearZ * Config::REGION_DIM_COUNT * Config::REGION_DIM_COUNT;
        fishIds[i] = i;
    }

    __global__ void findRegionStartsKernel(int* fishIds, int* regionIndexes, int* regionStarts) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i == 0) {
			regionStarts[regionIndexes[i]] = 0;
		}
		else if (regionIndexes[i] != regionIndexes[i - 1]) {
			regionStarts[regionIndexes[i]] = i;
		}
    }

    __global__ void computeMoveKernel(float* positions, FishShoalVelocities* velocities, FishProperties* properties)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int xIndex = i;
        int yIndex = i + Config::SHOAL_SIZE;
        int zIndex = i + Config::SHOAL_SIZE * 2;
        positions[xIndex] += velocities->velocityX[i];
        positions[yIndex] += velocities->velocityY[i];
        positions[zIndex] += velocities->velocityZ[i];
    }
}
