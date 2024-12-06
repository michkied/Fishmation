#include "computation/Kernels.h"
#include <limits>
#include <cmath>

namespace computation {
    __global__ void computeRegionsCheatSheetKernel(int* regionsCheatSheet)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= Config::REGION_COUNT) return;

        int dim = Config::REGION_DIM_COUNT;
		int x = i % dim;
		int y = (i / dim) % dim;
		int z = i / (dim * dim);

        int globalIndex = x * dim * dim + y * dim + z;
        int regionsToCheckIndex = 0;

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int k = -1; k <= 1; k++)
                {
                    int xIndex = x + i;
                    int yIndex = y + j;
                    int zIndex = z + k;

                    if (xIndex >= 0 && xIndex < dim && yIndex >= 0 && yIndex < dim && zIndex >= 0 && zIndex < dim)
                    {
                        regionsCheatSheet[globalIndex * 27 + regionsToCheckIndex] = xIndex * dim * dim + yIndex * dim + zIndex;
                        regionsToCheckIndex++;
                    }
                }
            }
        }

        for (int i = regionsToCheckIndex; i < 27; i++)
        {
            regionsCheatSheet[globalIndex * 27 + i] = -1;
        }
	}

    __global__ void assignFishToRegionsKernel(float* positions, int* fishIds, int* regionIndexes)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= Config::SHOAL_SIZE) return;
        
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

        int index = linearX + linearY * Config::REGION_DIM_COUNT + linearZ * Config::REGION_DIM_COUNT * Config::REGION_DIM_COUNT;
        regionIndexes[i] = index;
        regionIndexes[i + Config::SHOAL_SIZE] = index;  // this copy won't be sorted (for checking fish region in computeMoveKernel)
        fishIds[i] = i;
    }

    __global__ void findRegionStartsKernel(int* fishIds, int* regionIndexes, int* regionStarts) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= Config::SHOAL_SIZE) return;

		if (i == 0) {
			regionStarts[regionIndexes[i]] = 0;
		}
		else if (regionIndexes[i] != regionIndexes[i - 1]) {
			regionStarts[regionIndexes[i]] = i;
		}
    }

    __global__ void computeMoveKernel(float* positions, FishShoalVelocities* velocities, FishProperties* properties, int* fishIds, int* regionIndexes, int* regionStarts, int* regionsCheatSheet)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= Config::SHOAL_SIZE) return;

        int xIndex = i;
        int yIndex = i + Config::SHOAL_SIZE;
        int zIndex = i + Config::SHOAL_SIZE * 2;

        float Px = positions[xIndex];
        float Py = positions[yIndex];
        float Pz = positions[zIndex];

        float Vx = velocities->velocityX[i];
        float Vy = velocities->velocityY[i];
        float Vz = velocities->velocityZ[i];

        int numOfNeighbors = 0;

        float alignmentX = 0.0f;
        float alignmentY = 0.0f;
        float alignmentZ = 0.0f;

        float cohesionX = 0.0f;
        float cohesionY = 0.0f;
        float cohesionZ = 0.0f;

        float separationX = 0.0f;
        float separationY = 0.0f;
        float separationZ = 0.0f;

        int regionIndex = regionIndexes[i + Config::SHOAL_SIZE];  // use the copy of regionIndexes that is not sorted to get the region of the current fish

        // loop through neighboring regions
        int regionIterator = 0;
        int regionToCheck = regionsCheatSheet[regionIndex * 27 + regionIterator];
        while (regionToCheck != -1)
        {
            // loop through fish in the region
            int searchIndex = regionStarts[regionToCheck];
            while (searchIndex < Config::SHOAL_SIZE && regionIndexes[searchIndex] == regionToCheck) {
                int fishIndex = fishIds[searchIndex];
                if (fishIndex == i) {
                    searchIndex++;
                    continue;
                }

                float Qx = positions[fishIndex];
                float Qy = positions[fishIndex + Config::SHOAL_SIZE];
                float Qz = positions[fishIndex + Config::SHOAL_SIZE * 2];

                float distX = Qx - Px;
                float distY = Qy - Py;
                float distZ = Qz - Pz;

                float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
                if (dist < 0.000001f) dist = 0.000001f;
                if (dist > properties->viewDistance) {
                    searchIndex++;
                    continue;
                }

                float denom = sqrt(Vx * Vx + Vy * Vy + Vz * Vz) * dist;
                if (denom < 0.000001f) denom = 0.000001f;
                float cosAngle = (Vx * distX + Vy * distY + Vz * distZ) / denom;
                if (cosAngle < properties->fieldOfViewCos) {
                    searchIndex++;
                    continue;
                }

                alignmentX += velocities->velocityX[fishIndex];
                alignmentY += velocities->velocityY[fishIndex];
                alignmentZ += velocities->velocityZ[fishIndex];

                cohesionX += Qx;
                cohesionY += Qy;
                cohesionZ += Qz;

                separationX += distX / dist;
                separationY += distY / dist;
                separationZ += distZ / dist;

                numOfNeighbors++;
                searchIndex++;
            }

            regionIterator++;
            if (regionIterator == 27) break;
            regionToCheck = regionsCheatSheet[regionIndex * 27 + regionIterator];
		}

        if (numOfNeighbors > 0) {
            // Calculate alignment
            alignmentX = (alignmentX / numOfNeighbors - Vx) * properties->alignmentWeight;
            alignmentY = (alignmentY / numOfNeighbors - Vy) * properties->alignmentWeight;
            alignmentZ = (alignmentZ / numOfNeighbors - Vz) * properties->alignmentWeight;

            // Calculate cohesion
            cohesionX = ((cohesionX / numOfNeighbors) - Px) * properties->cohesionWeight;
            cohesionY = ((cohesionY / numOfNeighbors) - Py) * properties->cohesionWeight;
            cohesionZ = ((cohesionZ / numOfNeighbors) - Pz) * properties->cohesionWeight;

            // Calculate separation
            separationX = separationX / numOfNeighbors * properties->separationWeight;
            separationY = separationY / numOfNeighbors * properties->separationWeight;
            separationZ = separationZ / numOfNeighbors * properties->separationWeight;
		}

        // Calculate containment
        float containmentX = 0.0f;
        float containmentY = 0.0f;
        float containmentZ = 0.0f;

        float kF = properties->containmentWeight;

        float dist1X = Config::AQUARIUM_SIZE / 2 - Px;
        float dist2X = Config::AQUARIUM_SIZE / 2 + Px;
        containmentX -= kF / (dist1X * dist1X);
        containmentX += kF / (dist2X * dist2X);

        float dist1Y = Config::AQUARIUM_SIZE / 2 - Py;
        float dist2Y = Config::AQUARIUM_SIZE / 2 + Py;
        containmentY -= kF / (dist1Y * dist1Y);
        containmentY += kF / (dist2Y * dist2Y);

        float dist1Z = Config::AQUARIUM_SIZE / 2 - Pz;
        float dist2Z = Config::AQUARIUM_SIZE / 2 + Pz;
        containmentZ -= kF / (dist1Z * dist1Z);
        containmentZ += kF / (dist2Z * dist2Z);

        // Calculate net force
        float steeringX = alignmentX + cohesionX + separationX + containmentX;
        float steeringY = alignmentY + cohesionY + separationY + containmentY;
        float steeringZ = alignmentZ + cohesionZ + separationZ + containmentZ;

        float steeringStrength = sqrt(steeringX * steeringX + steeringY * steeringY + steeringZ * steeringZ);
        if (steeringStrength != 0.0f) {
            float clampedSteeringStrength = steeringStrength > properties->maxForce ? properties->maxForce : steeringStrength;

            float FsX = steeringX * clampedSteeringStrength / steeringStrength;
            float FsY = steeringY * clampedSteeringStrength / steeringStrength;
            float FsZ = steeringZ * clampedSteeringStrength / steeringStrength;

            float aX = FsX / properties->mass;
            float aY = FsY / properties->mass;
            float aZ = FsZ / properties->mass;

            Vx += aX;
            Vy += aY;
            Vz += aZ;
		}

        // Limit speed
        float speed = sqrt(Vx * Vx + Vy * Vy + Vz * Vz);
        if (speed > properties->maxSpeed) {
            Vx = Vx * properties->maxSpeed / speed;
            Vy = Vy * properties->maxSpeed / speed;
            Vz = Vz * properties->maxSpeed / speed;
        }

        float newPx = positions[xIndex] + Vx;
        if (Config::AQUARIUM_SIZE / 2 - newPx < 0 || Config::AQUARIUM_SIZE / 2 - newPx > 2) 
        {
			Vx = 0;
		}

        float newPy = positions[yIndex] + Vy;
        if (Config::AQUARIUM_SIZE / 2 - newPy < 0 || Config::AQUARIUM_SIZE / 2 - newPy > 2) 
		{
            Vy = 0;
        }

        float newPz = positions[zIndex] + Vz;
        if (Config::AQUARIUM_SIZE / 2 - newPz < 0 || Config::AQUARIUM_SIZE / 2 - newPz > 2)
        {
			Vz = 0;
		}

        positions[xIndex] += Vx;
        positions[yIndex] += Vy;
        positions[zIndex] += Vz;

        velocities->velocityX[i] = Vx;
        velocities->velocityY[i] = Vy;
        velocities->velocityZ[i] = Vz;
    }
}
