#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "fleet.h"

// basic, low overhead data structure for working with multiple data streams

Fleet * fleet_new(int num, int allocate)
{
    Fleet * nf = (Fleet *)malloc(sizeof(Fleet));
    memset(nf, 0, sizeof(Fleet));
    nf->fleet = (Motorist*)malloc(sizeof(Motorist) * num);
    nf->size = num;
    nf->streams = 0;

    assert(allocate <= num);
    fleet_preallocate(nf,allocate);
    return nf;
}

void fleet_destroy(Fleet * f)
{
    free(f->fleet);
    free(f);
}

void fleet_add_stream(Fleet * f,uint8_t * input, size_t isize, uint8_t * metadata, size_t msize, uint8_t * output, size_t osize)
{
    if (f->streams >= FLEET_CAPACITY)
    {
        fprintf(stderr,"error: reached capacity of %d\n", FLEET_CAPACITY);
        exit(1);
    }
    Motorist * m = f->fleet+f->streams;

    if (f->streams >= f->allocated)
    {
        motorist_init(m);
    }

    motorist_fuel(m,input,isize,metadata,msize);
    /*printf( "motorist ready? %d\n",m->phase == MotoristReady);*/
    m->output = output;
    f->streams++;
}

void fleet_preallocate(Fleet * f, int num)
{
    f->allocated = num;
    int i;
    for(i=0; i < num; i++)
    {
        motorist_init(f->fleet + i);
    }
}

Motorist * fleet_first(Fleet * f)
{
    return (f->mptr = (f->fleet+0));
}

Motorist * fleet_next(Fleet * f)
{
    return (f->mptr = f->mptr + 1);
}

uint8_t fleet_end(Fleet * f)
{
    return (f->mptr == (f->fleet + f->streams));
}



