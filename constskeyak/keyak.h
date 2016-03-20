#ifndef _KEYAK_H_
#define _KEYAK_H_

#include <stdint.h>
#include "motorist.h"
#include "engine.h"
#include "piston.h"


typedef struct _Keyak
{
    Buffer T;
    Buffer SUV;
    Motorist motorist;
    Buffer I,O,A;
} Keyak;


void keyak_init(Keyak* k);
void keyak_restart(Keyak * k);

void keyak_add_nonce(Keyak * k, uint8_t * nonce, uint32_t len);

void keyak_set_suv(Keyak * k, uint8_t * key, uint32_t klen);

void keyak_encrypt(Keyak * k, uint8_t * data, uint32_t datalen, 
                    uint8_t * metadata, uint32_t metalen);

void keyak_decrypt(Keyak * k, uint8_t * data, uint32_t datalen, 
                    uint8_t * metadata, uint32_t metalen, 
                    uint8_t * tag, uint32_t taglen);

#endif
