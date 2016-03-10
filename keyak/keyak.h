#ifndef _KEYAK_H_
#define _KEYAK_H_

#include <stdint.h>
#include "motorist.h"
#include "engine.h"
#include "piston.h"

typedef struct _KeyPack
{

} KeyPack;

typedef struct _Keyak
{
    uint32_t W;
    uint32_t c;
    Buffer T;
    Buffer SUV;
    Motorist motorist;
} Keyak;


void keyak_init(Keyak* k, uint32_t b, uint32_t nr, uint32_t c, uint32_t t);

#endif
