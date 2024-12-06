#pragma once

namespace Config {
	constexpr int WIDTH = 1920;
	constexpr int HEIGHT = 1080;
	constexpr int SHOAL_SIZE = 10000;
	constexpr int PREDATOR_COUNT = 5;
	constexpr float AQUARIUM_SIZE = 2.0f;
	constexpr float REGION_SIZE = 0.05f;

	constexpr float FISH_POINT_SIZE = 2.0f;
	constexpr float PREDATOR_POINT_SIZE = 5.0f;

	constexpr int THREADS_PER_BLOCK = 256;

	constexpr float PREDATOR_MAX_SPEED = 0.001f;
	constexpr float PREDATOR_MIN_SPEED = 0.0007f;

	constexpr int FISH_COUNT = SHOAL_SIZE + PREDATOR_COUNT;
	constexpr int REGION_DIM_COUNT = (int)(AQUARIUM_SIZE / REGION_SIZE);
	constexpr int REGION_COUNT = REGION_DIM_COUNT * REGION_DIM_COUNT * REGION_DIM_COUNT;
}