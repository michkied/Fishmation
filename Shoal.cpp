#pragma once
#include "Shoal.hpp"
#include <iostream>
#include <vector>
#include <chrono>

namespace graphics
{

    Shoal::Shoal(glm::mat4 view, glm::mat4 proj) {
        glEnable(GL_PROGRAM_POINT_SIZE);
        glGenVertexArrays(1, &_vao);
        glBindVertexArray(_vao);

        GLfloat vertices[] = {
            // Bottom face edges
            -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, // Top-left to Top-right
             0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f, // Top-right to Bottom-right
             0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, // Bottom-right to Bottom-left
            -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, // Bottom-left to Top-left

            // Top face edges
            -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, // Top-left to Top-right
             0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f, // Top-right to Bottom-right
             0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, // Bottom-right to Bottom-left
            -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, // Bottom-left to Top-left

            // Vertical edges
            -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, // Bottom Top-left to Top Top-left
             0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f, // Bottom Top-right to Top Top-right
             0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f, // Bottom Bottom-right to Top Bottom-right
            -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f  // Bottom Bottom-left to Top Bottom-left
        };

        glGenBuffers(1, &_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        CompileShaders();

        GLint uniView = glGetUniformLocation(_shaderProgram, "view");
        glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

        GLint uniProj = glGetUniformLocation(_shaderProgram, "proj");
        glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

        _uniModel = glGetUniformLocation(_shaderProgram, "model");

        _uniColor = glGetUniformLocation(_shaderProgram, "color");
        glUniform4f(_uniColor, 0.5f, 0.5f, 1.0f, 1.0f);

        GLint posAttrib = glGetAttribLocation(_shaderProgram, "position");
        glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(posAttrib);
    }

    Shoal::~Shoal() {
        glDeleteProgram(_shaderProgram);
        glDeleteShader(_fragmentShader);
        glDeleteShader(_vertexShader);

        glDeleteBuffers(1, &_vbo);
        glDeleteVertexArrays(1, &_vao);
    }

    void Shoal::CompileShaders() {
        CompileVertexShader();
        CompileGeometryShader();
        CompileFragmentShader();

        _shaderProgram = glCreateProgram();
        glAttachShader(_shaderProgram, _vertexShader);
        //glAttachShader(_shaderProgram, _geometryShader);
        glAttachShader(_shaderProgram, _fragmentShader);
        glBindFragDataLocation(_shaderProgram, 0, "outColor");

        glLinkProgram(_shaderProgram);
        glUseProgram(_shaderProgram);
    }

    void Shoal::CompileVertexShader() {
        const char* vertexSource = R"glsl(
            #version 150 core

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 proj;

            in vec3 position;

            void main()
            {
                gl_Position = proj * view * model * vec4(position, 1.0);
                gl_PointSize = 5.0;
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

    void Shoal::CompileGeometryShader() {
        const char* geometrySource = R"glsl(
            #version 150 core

            layout(points) in;
            layout(line_strip, max_vertices = 2) out;

            void main()
            {
                gl_Position = gl_in[0].gl_Position + vec4(-0.1, 0.0, 0.0, 0.0);
                EmitVertex();

                gl_Position = gl_in[0].gl_Position + vec4(0.1, 0.0, 0.0, 0.0);
                EmitVertex();

                EndPrimitive();
            }
        )glsl";

        _geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(_geometryShader, 1, &geometrySource, NULL);
        glCompileShader(_geometryShader);

        GLint status;
        glGetShaderiv(_geometryShader, GL_COMPILE_STATUS, &status);
        if (status != GL_TRUE)
        {
            GLint length;
            glGetShaderiv(_geometryShader, GL_INFO_LOG_LENGTH, &length);
            std::vector<char> log(length);
            glGetShaderInfoLog(_geometryShader, length, &length, log.data());
            std::cerr << "Geometry shader compilation failed: " << log.data() << std::endl;
            exit(1);
        }
    }

    void Shoal::CompileFragmentShader() {
        const char* fragmentSource = R"glsl(
			#version 150 core

            uniform vec4 color;

			out vec4 outColor;

			void main()
			{
				outColor = color;
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

    void Shoal::Draw(float time)
    {
        glBindVertexArray(_vao);
        glUseProgram(_shaderProgram);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(
            model,
            time * glm::radians(10.0f),
            glm::vec3(0.0f, 0.0f, 1.0f)
        );

        glUniformMatrix4fv(_uniModel, 1, GL_FALSE, glm::value_ptr(model));

        glDrawArrays(GL_POINTS, 0, 24);
    }
}
