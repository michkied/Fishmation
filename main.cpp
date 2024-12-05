#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thread>
#include <mutex>
#include <chrono>
#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "graphics/Aquarium.hpp"
#include "graphics/Shoal.hpp"
#include "Config.hpp"
#include "computation/Behavior.h"
#include "computation/types.h"

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

    float shoalData[Config::SHOAL_SIZE * 3] = {
            -0.9f, -0.91f, 0.0f, -0.5f, // x
            -0.9f, -0.8f,  0.0f, 0.5f, // y
            -0.9f, -0.5f,  0.0f, 0.5f  // z
    };

    glm::mat4 view = glm::lookAt(
        glm::vec3(3.0f, 0.0f, 0.5f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    glm::mat4 proj = glm::perspective(glm::radians(70.0f), 1920.0f / 1080.0f, 1.0f, 10.0f);

    computation::FishProperties properties;
    properties.mass = 1.0f;
    properties.maxForce = 1.0f;
    properties.maxSpeed = 0.001f;
    properties.fieldOfView = 180.0f;
    properties.viewDistance = 1.0f; // has to evenly divide the space
   

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
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}