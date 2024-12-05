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

        float containmentWeight;
        float alignmentWeight;
        float cohesionWeight;
        float separationWeight;
    };

    struct FishShoalVelocities
    {
        float velocityX[Config::SHOAL_SIZE];
        float velocityY[Config::SHOAL_SIZE];
        float velocityZ[Config::SHOAL_SIZE];
    };
}
