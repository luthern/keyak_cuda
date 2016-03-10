#ifndef _PISTON_H_
#define  _PISTON_H_

#include <stdint.h>
#include "defs.h"

typedef struct _Buffer
{
    uint32_t size;
    uint32_t offset;
    uint8_t buf[KEYAK_BUFFER_SIZE];
} Buffer;

typedef struct _Piston
{
    uint32_t Rs;
    uint32_t Ra;

    uint32_t EOM;
    uint32_t CryptEnd;
    uint32_t InjectStart;
    uint32_t InjectEnd;

    uint8_t state[KEYAK_STATE_SIZE+1];

} Piston;

void piston_init(Piston * p, uint32_t Rs, uint32_t Ra);

#endif
