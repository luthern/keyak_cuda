#include <cuda.h>
#include <stdio.h>
#include "utils.h"

void _HANDLE_ERROR(cudaError_t e, const char * file, int line)
{
    if (e != cudaSuccess)
    {
        printf("%s: %d. error %s\n", file, line, cudaGetErrorString(e));
        exit (1);
    }
}
