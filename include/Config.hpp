#pragma once

namespace Config {
	constexpr int WIDTH = 1920;
	constexpr int HEIGHT = 1080;
	constexpr int SHOAL_SIZE = 4;
	constexpr float AQUARIUM_SIZE = 2.0f;
	constexpr float REGION_SIZE = 1.0f;

	constexpr int REGION_DIM_COUNT = (int)(AQUARIUM_SIZE / REGION_SIZE);
	constexpr int REGION_COUNT = REGION_DIM_COUNT * REGION_DIM_COUNT * REGION_DIM_COUNT;
}