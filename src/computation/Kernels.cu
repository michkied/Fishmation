#include "computation/Kernels.h"
#include <cmath>
#include <limits>

namespace computation
{
	__global__ void computeRegionsCheatSheetKernel(Config* config, int* regionsCheatSheet)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= config->REGION_COUNT) return;

		int dim = config->REGION_DIM_COUNT;
		int x = i % dim;
		int y = (i / dim) % dim;
		int z = i / (dim * dim);

		int globalIndex = x * dim * dim + y * dim + z;
		int regionsToCheckIndex = 0;

		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				for (int k = -1; k <= 1; k++) {
					int xIndex = x + i;
					int yIndex = y + j;
					int zIndex = z + k;

					if (xIndex >= 0 && xIndex < dim && yIndex >= 0 && yIndex < dim && zIndex >= 0 && zIndex < dim) {
						regionsCheatSheet[globalIndex * 27 + regionsToCheckIndex] = xIndex * dim * dim + yIndex * dim + zIndex;
						regionsToCheckIndex++;
					}
				}
			}
		}

		for (int i = regionsToCheckIndex; i < 27; i++) {
			regionsCheatSheet[globalIndex * 27 + i] = -1;
		}
	}

	__global__ void assignFishToRegionsKernel(Config* config, float* positions, int* fishIds, int* regionIndexes)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= config->SHOAL_SIZE) return;

		int xIndex = i;
		int yIndex = i + config->FISH_COUNT;
		int zIndex = i + config->FISH_COUNT * 2;

		float tempX = positions[xIndex] / config->REGION_SIZE;
		float tempY = positions[yIndex] / config->REGION_SIZE;
		float tempZ = positions[zIndex] / config->REGION_SIZE;

		int idX = (tempX >= 0) ? (int)tempX + 1 : (int)tempX - 1;
		int idY = (tempY >= 0) ? (int)tempY + 1 : (int)tempY - 1;
		int idZ = (tempZ >= 0) ? (int)tempZ + 1 : (int)tempZ - 1;

		int linearX = config->REGION_DIM_COUNT / 2 - idX - (int)(idX < 0);
		int linearY = config->REGION_DIM_COUNT / 2 - idY - (int)(idY < 0);
		int linearZ = config->REGION_DIM_COUNT / 2 - idZ - (int)(idZ < 0);

		int index = linearX + linearY * config->REGION_DIM_COUNT + linearZ * config->REGION_DIM_COUNT * config->REGION_DIM_COUNT;
		regionIndexes[i] = index;
		regionIndexes[i + config->SHOAL_SIZE] = index;  // this copy won't be sorted (for checking fish region in computeMoveKernel)
		fishIds[i] = i;
	}

	__global__ void findRegionStartsKernel(Config* config, int* fishIds, int* regionIndexes, int* regionStarts)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= config->SHOAL_SIZE) return;

		if (i == 0) {
			regionStarts[regionIndexes[i]] = 0;
		}
		else if (regionIndexes[i] != regionIndexes[i - 1]) {
			regionStarts[regionIndexes[i]] = i;
		}
	}

	__global__ void computeShoalMoveKernel(Config* config, float* positions, float* velocities, FishProperties* properties, int* fishIds, int* regionIndexes, int* regionStarts, int* regionsCheatSheet)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= config->SHOAL_SIZE) return;

		int xIndex = i;
		int yIndex = i + config->FISH_COUNT;
		int zIndex = i + config->FISH_COUNT * 2;

		float Px = positions[xIndex];
		float Py = positions[yIndex];
		float Pz = positions[zIndex];

		float Vx = velocities[xIndex];
		float Vy = velocities[yIndex];
		float Vz = velocities[zIndex];

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
		float weightSum = 0.0f;

		int regionIndex = regionIndexes[i + config->SHOAL_SIZE];  // use the copy of regionIndexes that is not sorted to get the region of the current fish

		// loop through neighboring regions
		int regionIterator = 0;
		int regionToCheck = regionsCheatSheet[regionIndex * 27 + regionIterator];
		while (regionToCheck != -1) {
			// loop through fish in the region
			int searchIndex = regionStarts[regionToCheck];
			while (searchIndex < config->SHOAL_SIZE && regionIndexes[searchIndex] == regionToCheck) {
				int fishIndex = fishIds[searchIndex];
				if (fishIndex == i) {
					searchIndex++;
					continue;
				}

				float Qx = positions[fishIndex];
				float Qy = positions[fishIndex + config->FISH_COUNT];
				float Qz = positions[fishIndex + config->FISH_COUNT * 2];

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

				alignmentX += velocities[fishIndex];
				alignmentY += velocities[fishIndex + config->FISH_COUNT];
				alignmentZ += velocities[fishIndex + config->FISH_COUNT * 2];

				cohesionX += Qx;
				cohesionY += Qy;
				cohesionZ += Qz;

				separationX += -distX / dist;
				separationY += -distY / dist;
				separationZ += -distZ / dist;
				weightSum += 1.0f / dist;

				numOfNeighbors++;
				searchIndex++;
			}

			regionIterator++;
			if (regionIterator == 27) break;
			regionToCheck = regionsCheatSheet[regionIndex * 27 + regionIterator];
		}

		if (numOfNeighbors > 0) {
			// Calculate alignment
			alignmentX = (alignmentX / numOfNeighbors - Vx);
			alignmentY = (alignmentY / numOfNeighbors - Vy);
			alignmentZ = (alignmentZ / numOfNeighbors - Vz);

			float alignment = sqrt(alignmentX * alignmentX + alignmentY * alignmentY + alignmentZ * alignmentZ);
			if (alignment > 0.0f) {
				float kA = properties->alignmentWeight * config->ALIGNMENT_SCALE;
				alignmentX = alignmentX / alignment * kA;
				alignmentY = alignmentY / alignment * kA;
				alignmentZ = alignmentZ / alignment * kA;
			}

			// Calculate cohesion
			cohesionX = ((cohesionX / numOfNeighbors) - Px);
			cohesionY = ((cohesionY / numOfNeighbors) - Py);
			cohesionZ = ((cohesionZ / numOfNeighbors) - Pz);

			float cohesion = sqrt(cohesionX * cohesionX + cohesionY * cohesionY + cohesionZ * cohesionZ);
			if (cohesion > 0.0f) {
				float kC = properties->cohesionWeight * config->COHESION_SCALE;
				cohesionX = cohesionX / cohesion * kC;
				cohesionY = cohesionY / cohesion * kC;
				cohesionZ = cohesionZ / cohesion * kC;
			}

			// Calculate separation
			separationX = separationX / weightSum;
			separationY = separationY / weightSum;
			separationZ = separationZ / weightSum;

			float separation = sqrt(separationX * separationX + separationY * separationY + separationZ * separationZ);
			if (separation > 0.0f) {
				float kS = properties->separationWeight * config->SEPARATION_SCALE;
				separationX = separationX / separation * kS;
				separationY = separationY / separation * kS;
				separationZ = separationZ / separation * kS;
			}
		}

		// Calculate containment
		float containmentX = 0.0f;
		float containmentY = 0.0f;
		float containmentZ = 0.0f;

		float kF = properties->containmentWeight * config->CONTAINMENT_SCALE;

		float dist1X = config->AQUARIUM_SIZE / 2 - Px;
		float dist2X = config->AQUARIUM_SIZE / 2 + Px;
		containmentX -= kF / (dist1X * dist1X);
		containmentX += kF / (dist2X * dist2X);

		float dist1Y = config->AQUARIUM_SIZE / 2 - Py;
		float dist2Y = config->AQUARIUM_SIZE / 2 + Py;
		containmentY -= kF / (dist1Y * dist1Y);
		containmentY += kF / (dist2Y * dist2Y);

		float dist1Z = config->AQUARIUM_SIZE / 2 - Pz;
		float dist2Z = config->AQUARIUM_SIZE / 2 + Pz;
		containmentZ -= kF / (dist1Z * dist1Z);
		containmentZ += kF / (dist2Z * dist2Z);

		// Calculate predator avoidance
		float predatorAvoidanceX = 0.0f;
		float predatorAvoidanceY = 0.0f;
		float predatorAvoidanceZ = 0.0f;
		weightSum = 0.0f;

		for (int predatorIndex = config->SHOAL_SIZE; predatorIndex < config->FISH_COUNT; predatorIndex++) {
			float Qx = positions[predatorIndex];
			float Qy = positions[predatorIndex + config->FISH_COUNT];
			float Qz = positions[predatorIndex + config->FISH_COUNT * 2];

			float distX = Qx - Px;
			float distY = Qy - Py;
			float distZ = Qz - Pz;

			float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
			if (dist < 0.000001f) dist = 0.000001f;
			if (dist > properties->predatorViewDistance) continue;

			predatorAvoidanceX += -distX / dist;
			predatorAvoidanceY += -distY / dist;
			predatorAvoidanceZ += -distZ / dist;
			weightSum += 1.0f / dist;
		}

		float predatorAvoidance = sqrt(predatorAvoidanceX * predatorAvoidanceX + predatorAvoidanceY * predatorAvoidanceY + predatorAvoidanceZ * predatorAvoidanceZ);
		if (predatorAvoidance > 0.0f) {
			predatorAvoidanceX = predatorAvoidanceX / weightSum / predatorAvoidance * properties->predatorAvoidanceWeight;
			predatorAvoidanceY = predatorAvoidanceY / weightSum / predatorAvoidance * properties->predatorAvoidanceWeight;
			predatorAvoidanceZ = predatorAvoidanceZ / weightSum / predatorAvoidance * properties->predatorAvoidanceWeight;
		}

		// Calculate net force
		float FSx = alignmentX + cohesionX + separationX + containmentX + predatorAvoidanceX;
		float FSy = alignmentY + cohesionY + separationY + containmentY + predatorAvoidanceY;
		float FSz = alignmentZ + cohesionZ + separationZ + containmentZ + predatorAvoidanceZ;

		float force = sqrt(FSx * FSx + FSy * FSy + FSz * FSz);
		if (force != 0.0f) {
			float clampedFroce = force > properties->maxForce ? properties->maxForce : force;

			float Fx = FSx * clampedFroce / force;
			float Fy = FSy * clampedFroce / force;
			float Fz = FSz * clampedFroce / force;

			float aX = Fx / properties->mass;
			float aY = Fy / properties->mass;
			float aZ = Fz / properties->mass;

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

		float newPx = Px + Vx;
		if (config->AQUARIUM_SIZE / 2 - newPx < 0.001 || config->AQUARIUM_SIZE / 2 - newPx > 1.999) {
			Vx = 0;
		}

		float newPy = Py + Vy;
		if (config->AQUARIUM_SIZE / 2 - newPy < 0.001 || config->AQUARIUM_SIZE / 2 - newPy > 1.999) {
			Vy = 0;
		}

		float newPz = Pz + Vz;
		if (config->AQUARIUM_SIZE / 2 - newPz < 0.001 || config->AQUARIUM_SIZE / 2 - newPz > 1.999) {
			Vz = 0;
		}

		positions[xIndex] += Vx;
		positions[yIndex] += Vy;
		positions[zIndex] += Vz;

		velocities[xIndex] = Vx;
		velocities[yIndex] = Vy;
		velocities[zIndex] = Vz;
	}

	__global__ void computePredatorMoveKernel(Config* config, float* positions, float* velocities)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= config->PREDATOR_COUNT) return;

		int xIndex = i + config->SHOAL_SIZE;
		int yIndex = xIndex + config->FISH_COUNT;
		int zIndex = yIndex + config->FISH_COUNT;

		float Px = positions[xIndex];
		float Py = positions[yIndex];
		float Pz = positions[zIndex];

		float Vx = velocities[xIndex];
		float Vy = velocities[yIndex];
		float Vz = velocities[zIndex];

		float newPx = Px + Vx;
		if (config->AQUARIUM_SIZE / 2 - newPx < 0.001 || config->AQUARIUM_SIZE / 2 - newPx > 1.999) {
			Vx = -Vx;
		}

		float newPy = Py + Vy;
		if (config->AQUARIUM_SIZE / 2 - newPy < 0.001 || config->AQUARIUM_SIZE / 2 - newPy > 1.999) {
			Vy = -Vy;
		}

		float newPz = Pz + Vz;
		if (config->AQUARIUM_SIZE / 2 - newPz < 0.001 || config->AQUARIUM_SIZE / 2 - newPz > 1.999) {
			Vz = -Vz;
		}

		positions[xIndex] += Vx;
		positions[yIndex] += Vy;
		positions[zIndex] += Vz;

		velocities[xIndex] = Vx;
		velocities[yIndex] = Vy;
		velocities[zIndex] = Vz;
	}
}
