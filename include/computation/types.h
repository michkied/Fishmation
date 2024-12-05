#pragma once
#include "Config.hpp"

namespace computation {
    struct FishProperties
    {
        float mass;
        float maxForce;
        float maxSpeed;
        float fieldOfView;
        float viewDistance;
    };

    struct FishShoalVelocities
    {
        float velocityX[Config::SHOAL_SIZE];
        float velocityY[Config::SHOAL_SIZE];
        float velocityZ[Config::SHOAL_SIZE];
    };
}
