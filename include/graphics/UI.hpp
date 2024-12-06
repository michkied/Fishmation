#pragma once

#include "ui/imgui.h"
#include "ui/imgui_impl_glfw.h"
#include "ui/imgui_impl_opengl3.h"

#include "computation/types.h"

namespace graphics
{
	class UI
	{
	public:
		UI(GLFWwindow* window, computation::FishProperties& fishProperties, Config& config);
		~UI();

		void Draw();
	private:
		computation::FishProperties& _fishProperties;
		Config& _config;
		bool _displayPopup;
	};
}