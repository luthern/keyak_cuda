/*
    Author: Gerhard Hoffmann
    Basic implementation of Keccak on the GPU (NVIDIA)
    Model used: GTX 295
*/

#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <ctype.h>
#include <errno.h>
#include <cuda.h>
#include <stdlib.h>

//#include <shrUtils.h>
//#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include "cuda_keccak_basic.cuh"
#include <driver_types.h>

int main(int argc, const char** argv) {

    HANDLE_ERROR(cudaSetDevice(0));
    char * input = (char*) malloc(200);
    FILE *f = fopen("input", "rb");
    fread(input, 200, 1, f);
    fclose(f);
    int i;
    for (i = 0; i < 200; i++)
    {
        printf("%02hhx", input[i]);
    }
    printf("\n\n");
    PERMUTE((uint64_t*) input);
    for (i = 0; i < 200; i++)
    {
        printf("%02hhx", input[i]);
    }
    printf("\n\n");

    return 0;
}

/********************************** end-of-file ******************************/

