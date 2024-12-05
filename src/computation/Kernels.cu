#include "computation/Kernels.h"
#include <limits>
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

        float Px = positions[xIndex];
        float Py = positions[yIndex];
        float Pz = positions[zIndex];

        float Vx = velocities->velocityX[i];
        float Vy = velocities->velocityY[i];
        float Vz = velocities->velocityZ[i];

        float kF = 0.001f;

        float steeringX = 0.0f;
        float steeringY = 0.0f;
        float steeringZ = 0.0f;

        float dist1X = Config::AQUARIUM_SIZE / 2 - Px;
        float dist2X = -Config::AQUARIUM_SIZE / 2 - Px;
        steeringX -= kF * kF / (dist1X * dist1X);
        steeringX += kF * kF / (dist2X * dist2X);

        float dist1Y = Config::AQUARIUM_SIZE / 2 - Py;
        float dist2Y = -Config::AQUARIUM_SIZE / 2 - Py;
        steeringY -= kF * kF / (dist1Y * dist1Y);
        steeringY += kF * kF / (dist2Y * dist2Y);

        float dist1Z = Config::AQUARIUM_SIZE / 2 - Pz;
        float dist2Z = -Config::AQUARIUM_SIZE / 2 - Pz;
        steeringZ -= kF * kF / (dist1Z * dist1Z);
        steeringZ += kF * kF / (dist2Z * dist2Z);


  //      float timeToClosestWall = FLT_MAX;
  //      if (Vx > 0 && timeToClosestWall > (Config::AQUARIUM_SIZE / 2 - Px) / Vx) {
  //          timeToClosestWall = (Config::AQUARIUM_SIZE / 2 - Px) / Vx;
  //          steeringX = Vx - Vx * kF / timeToClosestWall / timeToClosestWall;
  //      }
  //      if (Vx < 0 && timeToClosestWall > (-Config::AQUARIUM_SIZE / 2 - Px) / Vx) {
		//	timeToClosestWall = (-Config::AQUARIUM_SIZE / 2 - Px) / Vx;
  //          steeringX = Vx - Vx * kF / timeToClosestWall / timeToClosestWall;
		//}
  //      if (Vy > 0 && timeToClosestWall > (Config::AQUARIUM_SIZE / 2 - Py) / Vy) {
  //          timeToClosestWall = (Config::AQUARIUM_SIZE / 2 - Py) / Vy;
  //          steeringX = Vx;
  //          steeringY = Vy - Vy * kF / timeToClosestWall / timeToClosestWall;
  //      }
  //      if (Vy < 0 && timeToClosestWall > (-Config::AQUARIUM_SIZE / 2 - Py) / Vy) {
		//	timeToClosestWall = (-Config::AQUARIUM_SIZE / 2 - Py) / Vy;
  //          steeringX = Vx;
		//	steeringY = Vy - Vy * kF / timeToClosestWall / timeToClosestWall;
		//}
  //      if (Vz > 0 && timeToClosestWall > (Config::AQUARIUM_SIZE / 2 - Pz) / Vz) {
		//	timeToClosestWall = (Config::AQUARIUM_SIZE / 2 - Pz) / Vz;
  //          steeringX = Vx;
  //          steeringY = Vy;
  //          steeringZ = Vz - Vz * kF / timeToClosestWall / timeToClosestWall;
		//}
  //      if (Vz < 0 && timeToClosestWall > (-Config::AQUARIUM_SIZE / 2 - Pz) / Vz) {
  //          timeToClosestWall = (-Config::AQUARIUM_SIZE / 2 - Pz) / Vz;
  //          steeringX = Vx;
  //          steeringY = Vy;
  //          steeringZ = Vz - Vz * kF / timeToClosestWall / timeToClosestWall;
  //      }

        float FsX = steeringX > properties->maxForce ? properties->maxForce : steeringX;
        float FsY = steeringY > properties->maxForce ? properties->maxForce : steeringY;
        float FsZ = steeringZ > properties->maxForce ? properties->maxForce : steeringZ;

        float aX = FsX / properties->mass;
        float aY = FsY / properties->mass;
        float aZ = FsZ / properties->mass;

        velocities->velocityX[i] = (Vx + aX > properties->maxSpeed) ? properties->maxSpeed : Vx + aX;
        velocities->velocityY[i] = (Vy + aY > properties->maxSpeed) ? properties->maxSpeed : Vy + aY;
        velocities->velocityZ[i] = (Vz + aZ > properties->maxSpeed) ? properties->maxSpeed : Vz + aZ;

        positions[xIndex] += velocities->velocityX[i];
        positions[yIndex] += velocities->velocityY[i];
        positions[zIndex] += velocities->velocityZ[i];


        //positions[xIndex] += velocities->velocityX[i];
        //positions[yIndex] += velocities->velocityY[i];
        //positions[zIndex] += velocities->velocityZ[i];
    }
}
