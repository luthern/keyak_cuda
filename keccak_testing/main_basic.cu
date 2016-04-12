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
#include "keccak.h"

void dump_hex(const char * str, uint8_t * bytes, uint8_t len)
{
    printf("%s\n",str);
    while(len--)
    {
        printf("%02hhx",*bytes++);
    }
    printf("\n\n");
}

int main(int argc, const char** argv) {

    HANDLE_ERROR(cudaSetDevice(0));
    int ec;
    uint8_t input[200];
    uint8_t input2[200];
    uint8_t input_copy[200];
    FILE *f = fopen("input", "rb");
    if (f == NULL)
    {
        fprintf(stderr,"input file \"input\" not found \n");
        exit(1);
    }
    ec = read(fileno(f), input, 200);
    if (ec != 200)
    {
        fprintf(stderr,"only read %d bytes\n",ec);
        exit(1);
    }
    fclose(f);

    memmove(input2,input,sizeof(input));
    memmove(input_copy,input,sizeof(input));
    dump_hex("input:",input,200);
    struct timer t;
    memset(&t, 0, sizeof(struct timer));

    gpu_init_keccak_tables();
    timer_start(&t, "1000000 sessions");
   
    int i;
    for (i = 0; i < 8; i++)
    {
        PERMUTE((uint64_t*) input);
        memmove(input,input_copy,sizeof(input));
    }
    PERMUTE((uint64_t*) input);
    cleanup_state();
    timer_end(&t);

    dump_hex("cuda keccak:",input,200);


    for (i = 0; i < 8; i++)
    {
        KeccakP1600_StatePermute(input2, 12, 0xd5);
        memmove(input2,input_copy,sizeof(input));
    }
    KeccakP1600_StatePermute(input2, 12, 0xd5);


    dump_hex("reference keccak:",input2,200);


    return 0;
}


