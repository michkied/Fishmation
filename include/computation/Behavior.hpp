
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace computation
{
    class Behavior
    {
    public:
  //      Behavior();
		//~Behavior();

        void Run();
    private:
        // Helper function for using CUDA to add vectors in parallel.
        cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
    };
}
