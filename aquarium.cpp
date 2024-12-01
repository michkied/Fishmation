#pragma once
#include "aquarium.hpp"
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

    Aquarium::Aquarium() {

        glGenVertexArrays(1, &_vao);
        glBindVertexArray(_vao);

        GLfloat vertices[] = {
            // Bottom face edges
            -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, // Top-left to Top-right
             1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f, // Top-right to Bottom-right
             1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, // Bottom-right to Bottom-left
            -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, // Bottom-left to Top-left

            // Top face edges
            -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, // Top-left to Top-right
             1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, // Top-right to Bottom-right
             1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, // Bottom-right to Bottom-left
            -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, // Bottom-left to Top-left

            // Vertical edges connecting top and bottom faces
            -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, // Bottom Top-left to Top Top-left
             1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f, // Bottom Top-right to Top Top-right
             1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, // Bottom Bottom-right to Top Bottom-right
            -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f  // Bottom Bottom-left to Top Bottom-left
        };

        glGenBuffers(1, &_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        //GLuint elements[] = {
        //    0, 1, 2,
        //};

        //glGenBuffers(1, &_ebo);
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
        //glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

        CompileShaders();
    }

    Aquarium::~Aquarium() {
        glDeleteProgram(_shaderProgram);
        glDeleteShader(_fragmentShader);
        glDeleteShader(_vertexShader);

        glDeleteBuffers(1, &_vbo);
        glDeleteVertexArrays(1, &_vao);
    }

    void Aquarium::CompileShaders() {
        CompileVertexShader();
        CompileFragmentShader();

        _shaderProgram = glCreateProgram();
        glAttachShader(_shaderProgram, _vertexShader);
        glAttachShader(_shaderProgram, _fragmentShader);
        glBindFragDataLocation(_shaderProgram, 0, "outColor");

        glLinkProgram(_shaderProgram);
        glUseProgram(_shaderProgram);

        GLint posAttrib = glGetAttribLocation(_shaderProgram, "position");
        glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(posAttrib);

        _uniColor = glGetUniformLocation(_shaderProgram, "triangleColor");
        glUniform4f(_uniColor, 1.0f, 1.0f, 1.0f, 1.0f);
    }

    void Aquarium::CompileVertexShader() {
        const char* vertexSource = R"glsl(
            #version 150 core

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 proj;

            in vec3 position;

            void main()
            {
                gl_Position = proj * view * model * vec4(position, 1.0);
            }
        )glsl";

        _vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(_vertexShader, 1, &vertexSource, NULL);
        glCompileShader(_vertexShader);

        GLint status;
        glGetShaderiv(_vertexShader, GL_COMPILE_STATUS, &status);
        if (status != GL_TRUE)
        {
            GLint length;
            glGetShaderiv(_vertexShader, GL_INFO_LOG_LENGTH, &length);
            std::vector<char> log(length);
            glGetShaderInfoLog(_vertexShader, length, &length, log.data());
            std::cerr << "Vertex shader compilation failed: " << log.data() << std::endl;
            exit(1);
        }
    }

    void Aquarium::CompileFragmentShader() {
        const char* fragmentSource = R"glsl(
			#version 150 core

            uniform vec4 triangleColor;

			out vec4 outColor;

			void main()
			{
				outColor = triangleColor;
			}
        )glsl";

        _fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(_fragmentShader, 1, &fragmentSource, NULL);
        glCompileShader(_fragmentShader);

        GLint status;
        glGetShaderiv(_fragmentShader, GL_COMPILE_STATUS, &status);
        if (status != GL_TRUE)
        {
            GLint length;
            glGetShaderiv(_fragmentShader, GL_INFO_LOG_LENGTH, &length);
            std::vector<char> log(length);
            glGetShaderInfoLog(_fragmentShader, length, &length, log.data());
            std::cerr << "Fragment shader compilation failed: " << log.data() << std::endl;
            exit(1);
        }
    }


    void Aquarium::Draw(float time)
    {

        glm::mat4 view = glm::lookAt(
            glm::vec3(3.0f, 0.0f, 0.5f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f)
        );
        glm::mat4 proj = glm::perspective(glm::radians(70.0f), static_cast<float>(WIDTH) / static_cast<float>(HEIGHT), 1.0f, 10.0f);

        GLint uniModel = glGetUniformLocation(_shaderProgram, "model");


        GLint uniView = glGetUniformLocation(_shaderProgram, "view");
        glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));


        GLint uniProj = glGetUniformLocation(_shaderProgram, "proj");
        glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(
            model,
            time * glm::radians(10.0f),
            glm::vec3(0.0f, 0.0f, 1.0f)
        );

        glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

        glDrawArrays(GL_LINES, 0, 24);

    }

    /*void Aquarium::Start() {

        GLint uniModel = glGetUniformLocation(_shaderProgram, "model");


        GLint uniView = glGetUniformLocation(_shaderProgram, "view");
        glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));


        GLint uniProj = glGetUniformLocation(_shaderProgram, "proj");
        glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

        auto t_start = std::chrono::high_resolution_clock::now();
        while (!glfwWindowShouldClose(_window)) {
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            auto t_now = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();

            glm::mat4 model = glm::mat4(1.0f);
            model = glm::rotate(
                model,
                time * glm::radians(10.0f),
                glm::vec3(0.0f, 0.0f, 1.0f)
            );

            glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

            glDrawArrays(GL_LINES, 0, 24);

            glfwSwapBuffers(_window);
            glfwPollEvents();
        }
    }*/
}
