
#include "computation/Kernel.h"

namespace computation {
    __global__ void addKernel(int* c, const int* a, const int* b)
    {
        int i = threadIdx.x;
        c[i] = a[i] + b[i];
    }
}
