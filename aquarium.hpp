

#include "glad/glad.h"
#include "GLFW/glfw3.h"

namespace graphics
{
	class Aquarium
	{
	public:
		Aquarium();
		~Aquarium();

		//void Start();
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
	};
}