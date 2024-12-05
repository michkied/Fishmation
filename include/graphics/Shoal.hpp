#pragma once

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace graphics
{
	class Shoal
	{
	public:
		Shoal(glm::mat4 view, glm::mat4 proj, float* shoalData);
		~Shoal();

		GLuint GetShoalBuffer();
		void Draw(float time);

	private:
		void CompileShaders();
		void CompileVertexShader();
		void CompileGeometryShader();
		void CompileFragmentShader();

		float* _shoalData;

		GLuint _shaderProgram;
		GLuint _vao;
		GLuint _vbo;
		GLuint _vertexShader;
		GLuint _geometryShader;
		GLuint _fragmentShader;

		GLint _uniColor;
		GLint _uniModel;
	};
}