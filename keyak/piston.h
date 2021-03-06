#ifndef _PISTON_H_
#define  _PISTON_H_

#include <stdint.h>
#include <string.h>
#include "defs.h"

typedef struct _Buffer
{
    uint32_t offset;
    uint32_t length;
    uint8_t * buf;
    uint8_t buf_stack[KEYAK_BUFFER_SIZE];
} Buffer;

#define buffer_put(b,d)     ( (b)->buf[(b)->length++] = (d) )
#define buffer_get(b)       ( (b)->buf[(b)->offset++] )
#define buffer_has_more(b)  ( (b)->offset < (b)->length)
#define buffer_seek(b,d)    ( (b)->offset = (d) )
#define buffer_same(b1,b2)  ( memcmp((b1)->buf,(b2)->buf, KEYAK_BUFFER_SIZE) == 0 )
#define buffer_clone(b1,b2) ( buffer_init((b1), (b2)->buf, (b2)->length) )


void buffer_init(Buffer * b, uint8_t * data, uint32_t len);

#endif
