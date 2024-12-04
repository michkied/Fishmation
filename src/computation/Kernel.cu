
#include "computation/Kernel.h"

namespace computation {
    __global__ void computeMoveKernel(float* positions)
    {
        int i = threadIdx.x;
        positions[i + 6] += 0.001f;
    }
}
