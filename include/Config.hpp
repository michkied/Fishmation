#pragma once

namespace Config {
	constexpr int WIDTH = 1920;
	constexpr int HEIGHT = 1080;
	constexpr int SHOAL_SIZE = 10000;
	constexpr float AQUARIUM_SIZE = 2.0f;
	constexpr float REGION_SIZE = 0.05f;

	constexpr int THREADS_PER_BLOCK = 256;

	constexpr int REGION_DIM_COUNT = (int)(AQUARIUM_SIZE / REGION_SIZE);
	constexpr int REGION_COUNT = REGION_DIM_COUNT * REGION_DIM_COUNT * REGION_DIM_COUNT;
}