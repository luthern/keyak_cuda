#ifndef _PISTON_H_
#define  _PISTON_H_

#include <stdint.h>
#include <string.h>
#include "defs.h"

typedef struct _Buffer
{
    uint32_t offset;
    uint32_t length;
    uint8_t buf[KEYAK_BUFFER_SIZE];
} Buffer;

#define buffer_put(b,d)     ( (b)->buf[(b)->length++] = (d) )
#define buffer_get(b)       ( (b)->buf[(b)->offset++] )
#define buffer_has_more(b)  ( (b)->offset < (b)->length)
#define buffer_seek(b,d)    ( (b)->offset = (d) )
#define buffer_same(b1,b2)  ( memcmp((b1)->buf,(b2)->buf, KEYAK_BUFFER_SIZE) == 0 )
#define buffer_clone(b1,b2) ( buffer_init((b1), (b2)->buf, (b2)->length) )


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

void buffer_init(Buffer * b, uint8_t * data, uint32_t len);

void piston_init(Piston * p);
void piston_get_tag(Piston * p, Buffer * T, uint32_t l);
//__global__ void piston_inject(Piston * p, Buffer * x, uint8_t crypting);
void piston_spark(Piston * p, uint8_t eom, uint8_t offset);
//void piston_crypt(Piston * p, Buffer * I, Buffer * O, uint8_t w,
//        uint8_t unwrapFlag);


void piston_restart(Piston * p);

#endif
