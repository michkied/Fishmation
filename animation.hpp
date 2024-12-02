
#include "aquarium.hpp"
#include "Shoal.hpp"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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

		glm::mat4 view = glm::lookAt(
			glm::vec3(3.0f, 0.0f, 0.5f),
			glm::vec3(0.0f, 0.0f, 0.0f),
			glm::vec3(0.0f, 0.0f, 1.0f)
		);
		glm::mat4 proj = glm::perspective(glm::radians(70.0f), 1920.0f / 1080.0f, 1.0f, 10.0f);

		Aquarium _aquarium = Aquarium(view, proj);
		Shoal _shoal = Shoal(view, proj);

	};
}