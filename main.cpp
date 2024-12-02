#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thread>
#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "graphics/Animation.hpp"
#include "Config.hpp"
#include "computation/Behavior.hpp"

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

    float shoal[Config::SHOAL_SIZE * 3] = {
            -0.5f,  -0.5f, -0.5f,
            -0.5f,  -0.5f,  0.5f,
            -0.5f,  0.5f,  0.5f,
            -0.5f,  0.5f, -0.5f,
            0.5f,  -0.5f, -0.5f,
            0.5f,  -0.5f,  0.5f,
            0.5f,  0.5f,  0.5f,
            0.5f,  0.5f, -0.5f,
    };

    computation::Behavior behavior = computation::Behavior();
    std::thread behaviorThread(&computation::Behavior::Run, &behavior);

    graphics::Animation animation = graphics::Animation(window, shoal);
    animation.Start();

    behaviorThread.join();

    return 0;
}