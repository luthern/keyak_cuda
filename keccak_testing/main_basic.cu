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
#include "misc.h"

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
    struct timer t;
    memset(&t, 0, sizeof(struct timer));

    gpu_init_keccak_tables();
    timer_start(&t, "1000000 sessions");
    
    PERMUTE((uint64_t*) input);
    cleanup_state();
    timer_end(&t);

    for (i = 0; i < 200; i++)
    {
        printf("%02hhx", input[i]);
    }
    printf("\n\n");

    return 0;
}

/********************************** end-of-file ******************************/

