#include "graphics/UI.hpp"

namespace graphics
{
	UI::UI(GLFWwindow* window, computation::FishProperties& fishProperties, Config& config) : _fishProperties(fishProperties), _config(config), _displayPopup(!config.loadConfigSuccess)
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ImGui::StyleColorsDark();

		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 330");
	}

	UI::~UI()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	void UI::Draw()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Fishmation");
		if (_config.FULLSCREEN) {
			ImGui::Text("Running in fullscreen mode. Press ESC to exit.");
		}

		ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

		ImGui::Text("Predator detection distance");
		ImGui::SliderFloat("##predatorDect", &_fishProperties.predatorViewDistance, 0.0f, 1.2f);
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::Text("Containment");
		ImGui::SliderFloat("##containment", &_fishProperties.containmentWeight, 0.0f, 100.0f, "%.1f");
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::Text("Alignment");
		ImGui::SliderFloat("##alignment", &_fishProperties.alignmentWeight, 0.0f, 100.0f, "%.1f");
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::Text("Cohesion");
		ImGui::SliderFloat("##cohesion", &_fishProperties.cohesionWeight, 0.0f, 100.0f, "%.1f");
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::Text("Separation");
		ImGui::SliderFloat("##separation", &_fishProperties.separationWeight, 0.0f, 100.0f, "%.1f");
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::End();

		if (_displayPopup) {
			ImGui::OpenPopup("Warning");
		}

		if (ImGui::BeginPopup("Warning")) {
			ImGui::Text("The config.txt file was not found. The program will run on its default configuration.");
			ImGui::Separator();
			if (ImGui::Button("Close")) {
				ImGui::CloseCurrentPopup();
				_displayPopup = false;
			}
			ImGui::EndPopup();
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}