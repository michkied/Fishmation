#include "Config.hpp"

#include <fstream>
#include <sstream>

Config::Config()
{
	std::ifstream file("config.txt");
	if (!file) {
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream is_line(line);
		std::string key, value;
		if (!std::getline(is_line, key, '=')) continue;
		if (!std::getline(is_line, value)) continue;

		if (key == "FULLSCREEN" && (value == "TRUE" || value == "true")) FULLSCREEN = true;
		else if (key == "WIDTH") WIDTH = std::stoi(value);
		else if (key == "HEIGHT") HEIGHT = std::stoi(value);
		else if (key == "SHOAL_SIZE") SHOAL_SIZE = std::stoi(value);
		else if (key == "FISH_POINT_SIZE") FISH_POINT_SIZE = std::stof(value);
		else if (key == "PREDATOR_COUNT") PREDATOR_COUNT = std::stoi(value);
		else if (key == "PREDATOR_POINT_SIZE") PREDATOR_POINT_SIZE = std::stof(value);
		else if (key == "PREDATOR_MAX_SPEED") PREDATOR_MAX_SPEED = std::stof(value);
		else if (key == "PREDATOR_MIN_SPEED") PREDATOR_MIN_SPEED = std::stof(value);
		else if (key == "AQUARIUM_SIZE") AQUARIUM_SIZE = std::stof(value);
		else if (key == "REGION_SIZE") REGION_SIZE = std::stof(value);
		else if (key == "THREADS_PER_BLOCK") THREADS_PER_BLOCK = std::stoi(value);
		else if (key == "CONTAINMENT_SCALE") CONTAINMENT_SCALE = std::stof(value);
		else if (key == "ALIGNMENT_SCALE") ALIGNMENT_SCALE = std::stof(value);
		else if (key == "COHESION_SCALE") COHESION_SCALE = std::stof(value);
		else if (key == "SEPARATION_SCALE") SEPARATION_SCALE = std::stof(value);
	}

	FISH_COUNT = SHOAL_SIZE + PREDATOR_COUNT;
	REGION_DIM_COUNT = (int)(AQUARIUM_SIZE / REGION_SIZE);
	REGION_COUNT = REGION_DIM_COUNT * REGION_DIM_COUNT * REGION_DIM_COUNT;

	loadConfigSuccess = true;
}