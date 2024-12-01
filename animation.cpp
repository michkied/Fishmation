#pragma once
#include "animation.hpp"
#include <iostream>
#include <vector>
#include <chrono>

#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace graphics 
{
    constexpr int WIDTH = 1920;
    constexpr int HEIGHT = 1080;

    Animation::Animation(GLFWwindow* window) : _window(window) {}

    Animation::~Animation() {

        glfwDestroyWindow(_window);
        glfwTerminate();
	}

    void Animation::Start() {

        auto t_start = std::chrono::high_resolution_clock::now();
        while (!glfwWindowShouldClose(_window)) {
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            auto t_now = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();

            _aquarium.Draw(time);

            glfwSwapBuffers(_window);
            glfwPollEvents();
        }
	}
}
