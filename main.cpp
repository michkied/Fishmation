#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <sstream>
#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "graphics/Aquarium.hpp"
#include "graphics/Shoal.hpp"
#include "Config.hpp"
#include "computation/Behavior.h"
#include "computation/types.h"

void update_fps_counter(GLFWwindow* window) {
    static double previous_seconds = glfwGetTime();
    static int frame_count = 0;

    double current_seconds = glfwGetTime();
    double elapsed_seconds = current_seconds - previous_seconds;

    if (elapsed_seconds > 1.0) {
        previous_seconds = current_seconds;
        double fps = (double)frame_count / elapsed_seconds;

        std::ostringstream oss;
        oss.precision(2);
        oss << std::fixed << "Fishmation - FPS: " << fps;
        glfwSetWindowTitle(window, oss.str().c_str());

        frame_count = 0;
    }

    frame_count++;
}

int main()
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* window = glfwCreateWindow(Config::WIDTH, Config::HEIGHT, "GLFW OpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0f, 0.5f);

    float* shoalData = new float[Config::FISH_COUNT * 3];
    for (int i = 0; i < Config::FISH_COUNT * 3; i++) {
        float random_number;
        do { random_number = distribution(gen); } 
        while (random_number < -0.9f || random_number > 0.9f);
		shoalData[i] = random_number;
	}

    glm::mat4 view = glm::lookAt(
        glm::vec3(3.0f, 0.0f, 0.5f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    glm::mat4 proj = glm::perspective(glm::radians(70.0f), 1920.0f / 1080.0f, 1.0f, 10.0f);

    computation::FishProperties properties;
    properties.mass = 1.0f;
    properties.maxForce = 0.0001f;
    properties.maxSpeed = 0.001f;
    properties.fieldOfViewCos = (float)std::cos(180.0f / 2 * 3.14159 / 180);
    properties.viewDistance = Config::REGION_SIZE;
    properties.predatorViewDistance = Config::REGION_SIZE * 7;

    properties.containmentWeight = 1.0f / 10000000.0f;
    properties.alignmentWeight = 0.01f;
    properties.cohesionWeight = 0.001f;
    properties.separationWeight = 0.005f;
    properties.predatorAvoidanceWeight = 5.0f;

    graphics::Aquarium aquarium = graphics::Aquarium(view, proj);
    graphics::Shoal shoal = graphics::Shoal(view, proj, shoalData);
    computation::Behavior behavior = computation::Behavior(shoal.GetShoalBuffer(), properties);

    cudaError_t cudaStatus;
    auto t_start = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto t_now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();

        aquarium.Draw(time);
        shoal.Draw(time);

        glfwSwapBuffers(window);
        glfwPollEvents();

        cudaStatus = behavior.ComputeMove();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "ComputeMove failed!");
            return 1;
        }

        update_fps_counter(window);
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}