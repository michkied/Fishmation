#include "graphics/UI.hpp"

namespace graphics {
	UI::UI(GLFWwindow* window, computation::FishProperties& fishProperties) : _fishProperties(fishProperties) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
	}

    UI::~UI() {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

    void UI::Draw() {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Fishmation");
		ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

        ImGui::Text("Predator detection distance");
        ImGui::SliderFloat("##predatorDect", &_fishProperties.predatorViewDistance, 0.0f, 1.0f);
        if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

        ImGui::Text("Predator avoidance");
		ImGui::SliderFloat("##predator", &_fishProperties.predatorAvoidanceWeight, 0.0f, 0.01f, "%.5f");
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::Text("Containment");
		ImGui::SliderFloat("##containment", &_fishProperties.containmentWeight, 0.0f, 0.01f, "%.5f");
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::Text("Alignment");
		ImGui::SliderFloat("##alignment", &_fishProperties.alignmentWeight, 0.0f, 0.01f, "%.5f");
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::Text("Cohesion");
		ImGui::SliderFloat("##cohesion", &_fishProperties.cohesionWeight, 0.0f, 0.01f, "%.5f");
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::Text("Separation");
		ImGui::SliderFloat("##separation", &_fishProperties.separationWeight, 0.0f, 0.01f, "%.5f");
		if (ImGui::IsItemActive()) {
			_fishProperties.changeCounter++;
		}

		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        //// Start ImGui frame
        //ImGui_ImplOpenGL3_NewFrame();
        //ImGui_ImplGlfw_NewFrame();
        //ImGui::NewFrame();



        //// Render ImGui
        //ImGui::Render();
        //int display_w, display_h;
        //glfwGetFramebufferSize(window, &display_w, &display_h);
        //glViewport(0, 0, display_w, display_h);
        //ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}