#pragma once
#include "Config.hpp"

namespace computation {
    struct FishProperties
    {
        double mass;
        double maxForce;
        double maxSpeed;
        double fieldOfView;
        double viewDistance;
    };

    struct FishShoal
    {
        double positionX[Config::SHOAL_SIZE];
        double positionY[Config::SHOAL_SIZE];
        double positionZ[Config::SHOAL_SIZE];

        double velocityX[Config::SHOAL_SIZE];
        double velocityY[Config::SHOAL_SIZE];
        double velocityZ[Config::SHOAL_SIZE];
    };
}
