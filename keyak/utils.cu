#include <cuda.h>
#include "utils.h"

void _HANDLE_ERROR(cudaError_t e, int line)
{
    if (e != cudaSuccess)
    {
        printf("line: %d. error %s\n", line, cudaGetErrorString(e));
        exit (1);
    }
}
