#ifndef _FLEET_H_
#define _FLEET_H_

#include "motorist.h"

#define FLEET_CAPACITY  10

static Motorist fleet[FLEET_CAPACITY];
static int num_boats = 0;
static int num_allocated= 0;
static Motorist * mptr = NULL;

typedef struct _Fleet
{
    int size;
    int streams;
    int allocated;
    Motorist * mptr;
    Motorist * fleet;
} Fleet;

Fleet * fleet_new(int num, int allocate);

void fleet_add_stream(Fleet * f,uint8_t * input, size_t isize, uint8_t * metadata, size_t msize, uint8_t * output, size_t osize, uint8_t * tag);

void fleet_preallocate(Fleet * f, int num);

Motorist * fleet_first(Fleet * f);
Motorist * fleet_next(Fleet * f);
uint8_t fleet_end(Fleet * f);
void fleet_destroy(Fleet * f);



#endif
