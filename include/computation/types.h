#pragma once
#include "Config.hpp"

namespace computation
{
	struct FishProperties
	{
		float mass;
		float maxForce;
		float maxSpeed;
		float fieldOfViewCos;
		float viewDistance;
		float predatorViewDistance;

		float containmentWeight;
		float alignmentWeight;
		float cohesionWeight;
		float separationWeight;
		float predatorAvoidanceWeight;

		int changeCounter = 0;
	};
}
