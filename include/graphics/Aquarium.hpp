#pragma once

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace graphics
{
	class Aquarium
	{
	public:
		Aquarium(glm::mat4 view, glm::mat4 proj);
		~Aquarium();

		void Draw(float time);

	private:
		void CompileShaders();
		void CompileVertexShader();
		void CompileFragmentShader();

		GLFWwindow* _window;
		GLuint _shaderProgram;
		GLuint _vao;
		GLuint _vbo;
		GLuint _vertexShader;
		GLuint _fragmentShader;

		GLint _uniColor;
		GLint _uniModel;
	};
}