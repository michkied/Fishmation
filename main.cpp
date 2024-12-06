#include "cuda_runtime.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include <chrono>
#include <cmath>
#include <mutex>
#include <random>

#include "computation/Behavior.h"
#include "computation/types.h"
#include "Config.hpp"
#include "graphics/Aquarium.hpp"
#include "graphics/Shoal.hpp"
#include "graphics/UI.hpp"

void quit(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

int main()
{
	Config config = Config();

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	GLFWwindow* window = glfwCreateWindow(config.WIDTH, config.HEIGHT, "Fishmation - Loading...", config.FULLSCREEN ? glfwGetPrimaryMonitor() : NULL, NULL);
	glfwMakeContextCurrent(window);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	if (config.FULLSCREEN) {
		glfwSetKeyCallback(window, quit);
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);

	// Generate random fish positions from a normal distribution
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> distribution(0.0f, 0.5f);
	float* shoalData = new float[config.FISH_COUNT * 3];
	for (int i = 0; i < config.FISH_COUNT * 3; i++) {
		float random_number;
		do { random_number = distribution(gen); } while (random_number < -0.9f || random_number > 0.9f);
		shoalData[i] = random_number;
	}

	// Generate perspective matrices
	glm::mat4 view = glm::lookAt(
		glm::vec3(3.0f, 0.0f, 0.5f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f)
	);
	glm::mat4 proj = glm::perspective(glm::radians(70.0f), (float)config.WIDTH / (float)config.HEIGHT, 1.0f, 10.0f);

	// Set initial fish properties
	computation::FishProperties properties;
	properties.mass = 1.0f;
	properties.maxForce = 0.0001f;
	properties.maxSpeed = 0.001f;
	properties.fieldOfViewCos = (float)std::cos(270.0f / 2 * 3.14159 / 180);
	properties.viewDistance = config.REGION_SIZE;

	properties.predatorViewDistance = 0.6f;
	properties.containmentWeight = 50.0f;
	properties.alignmentWeight = 50.0f;
	properties.cohesionWeight = 50.0f;
	properties.separationWeight = 50.0f;

	properties.predatorAvoidanceWeight = 5.0f;

	graphics::UI ui = graphics::UI(window, properties, config);
	graphics::Aquarium aquarium = graphics::Aquarium(view, proj);
	graphics::Shoal shoal = graphics::Shoal(config, view, proj, shoalData);
	computation::Behavior behavior = computation::Behavior(config, shoal.GetShoalBuffer(), properties);
	delete[] shoalData;

	glfwSetWindowTitle(window, "Fishmation");

	cudaError_t cudaStatus;
	auto t_start = std::chrono::high_resolution_clock::now();
	while (!glfwWindowShouldClose(window)) {
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		auto t_now = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();

		aquarium.Draw(time);
		shoal.Draw(time);
		ui.Draw();

		glfwSwapBuffers(window);
		glfwPollEvents();

		cudaStatus = behavior.ComputeMove();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "ComputeMove failed!");
			return 1;
		}
	}

	return 0;
}