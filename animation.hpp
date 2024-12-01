
#include "aquarium.hpp"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

namespace graphics 
{
	class Animation
	{
	public:
		Animation(GLFWwindow* window);
		~Animation();

		void Start();

	private:
		GLFWwindow* _window;
		Aquarium _aquarium;

	};
}