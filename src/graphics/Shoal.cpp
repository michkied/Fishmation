#include "Config.hpp"
#include "graphics/Shoal.hpp"
#include <chrono>
#include <iostream>
#include <vector>

namespace graphics
{

	Shoal::Shoal(Config& config, glm::mat4 view, glm::mat4 proj, float* shoalData) : _shoalData(shoalData), _config(config)
	{
		glEnable(GL_PROGRAM_POINT_SIZE);
		glGenVertexArrays(1, &_vao);
		glBindVertexArray(_vao);

		glGenBuffers(1, &_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, _vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * config.FISH_COUNT * 3, shoalData, GL_STREAM_DRAW);

		CompileShaders();

		GLint uniView = glGetUniformLocation(_shaderProgram, "view");
		glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

		GLint uniProj = glGetUniformLocation(_shaderProgram, "proj");
		glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

		_uniModel = glGetUniformLocation(_shaderProgram, "model");
		_uniColor = glGetUniformLocation(_shaderProgram, "color");
		_uniSize = glGetUniformLocation(_shaderProgram, "size");

		GLint posXAttrib = glGetAttribLocation(_shaderProgram, "posX");
		glVertexAttribPointer(posXAttrib, 1, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(posXAttrib);

		GLint posYAttrib = glGetAttribLocation(_shaderProgram, "posY");
		glVertexAttribPointer(posYAttrib, 1, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(sizeof(GLfloat) * config.FISH_COUNT));
		glEnableVertexAttribArray(posYAttrib);

		GLint posZAttrib = glGetAttribLocation(_shaderProgram, "posZ");
		glVertexAttribPointer(posZAttrib, 1, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(sizeof(GLfloat) * config.FISH_COUNT * 2));
		glEnableVertexAttribArray(posZAttrib);
	}

	Shoal::~Shoal()
	{
		glDeleteProgram(_shaderProgram);
		glDeleteShader(_fragmentShader);
		glDeleteShader(_vertexShader);

		glDeleteBuffers(1, &_vbo);
		glDeleteVertexArrays(1, &_vao);
	}

	GLuint Shoal::GetShoalBuffer()
	{
		return _vbo;
	}

	void Shoal::CompileShaders()
	{
		CompileVertexShader();
		CompileFragmentShader();

		_shaderProgram = glCreateProgram();
		glAttachShader(_shaderProgram, _vertexShader);
		glAttachShader(_shaderProgram, _fragmentShader);
		glBindFragDataLocation(_shaderProgram, 0, "outColor");

		glLinkProgram(_shaderProgram);
		glUseProgram(_shaderProgram);
	}

	void Shoal::CompileVertexShader()
	{
		const char* vertexSource = R"glsl(
            #version 150 core

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 proj;
            uniform float size;

            in float posX;
            in float posY;
			in float posZ;

            void main()
            {
                gl_Position = proj * view * model * vec4(posX, posY, posZ, 1.0);
                gl_PointSize = size;
            }
        )glsl";

		_vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(_vertexShader, 1, &vertexSource, NULL);
		glCompileShader(_vertexShader);

		GLint status;
		glGetShaderiv(_vertexShader, GL_COMPILE_STATUS, &status);
		if (status != GL_TRUE) {
			GLint length;
			glGetShaderiv(_vertexShader, GL_INFO_LOG_LENGTH, &length);
			std::vector<char> log(length);
			glGetShaderInfoLog(_vertexShader, length, &length, log.data());
			std::cerr << "Vertex shader compilation failed: " << log.data() << std::endl;
			exit(1);
		}
	}

	void Shoal::CompileFragmentShader()
	{
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
		if (status != GL_TRUE) {
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

		glUniform1f(_uniSize, _config.FISH_POINT_SIZE);
		glUniform4f(_uniColor, 0.5f, 0.5f, 1.0f, 1.0f);
		glDrawArrays(GL_POINTS, 0, _config.SHOAL_SIZE);

		glUniform1f(_uniSize, _config.PREDATOR_POINT_SIZE);
		glUniform4f(_uniColor, 1.0f, 0.0f, 0.0f, 1.0f);
		glDrawArrays(GL_POINTS, _config.SHOAL_SIZE, _config.PREDATOR_COUNT);
	}
}
