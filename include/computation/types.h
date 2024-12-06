#pragma once
#include "Config.hpp"

namespace computation {
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

    struct FishShoalVelocities
    {
        float velocityX[Config::SHOAL_SIZE];
        float velocityY[Config::SHOAL_SIZE];
        float velocityZ[Config::SHOAL_SIZE];
    };

    struct PredatorVelocities
    {
        float velocityX[Config::PREDATOR_COUNT];
        float velocityY[Config::PREDATOR_COUNT];
        float velocityZ[Config::PREDATOR_COUNT];
    };
}
