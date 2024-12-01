#pragma once
#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace graphics
{
    constexpr int WIDTH = 1920;
    constexpr int HEIGHT = 1080;

    glm::mat4 View = glm::lookAt(
        glm::vec3(3.0f, 0.0f, 0.5f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    glm::mat4 Proj = glm::perspective(glm::radians(70.0f), static_cast<float>(WIDTH) / static_cast<float>(HEIGHT), 1.0f, 10.0f);
}