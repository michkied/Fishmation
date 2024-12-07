#pragma once

class Config
{
public:
	Config();

	bool loadConfigSuccess = false;

	bool FULLSCREEN = false;
	int WIDTH = 1920;
	int HEIGHT = 1080;

	int SHOAL_SIZE = 10000;
	float FISH_POINT_SIZE = 2.0f;

	int PREDATOR_COUNT = 5;
	float PREDATOR_POINT_SIZE = 5.0f;
	float PREDATOR_MAX_SPEED = 0.001f;
	float PREDATOR_MIN_SPEED = 0.0007f;

	float REGION_SIZE = 0.01f; // should divide the aquarium evenly

	int THREADS_PER_BLOCK = 256;

	float CONTAINMENT_SCALE = 1.0f / 30000000;
	float ALIGNMENT_SCALE = 1.0f;
	float COHESION_SCALE = 1.0f / 2;
	float SEPARATION_SCALE = 1.0f;

	// Do not change
	const float AQUARIUM_SIZE = 2.0f;
	int FISH_COUNT = SHOAL_SIZE + PREDATOR_COUNT;
	int REGION_DIM_COUNT = (int)(AQUARIUM_SIZE / REGION_SIZE);
	int REGION_COUNT = REGION_DIM_COUNT * REGION_DIM_COUNT * REGION_DIM_COUNT;
};