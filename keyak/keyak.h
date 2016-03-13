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
    
    Buffer I,O,A;

} Keyak;


void keyak_init(Keyak* k, uint32_t b, uint32_t nr, uint32_t c, uint32_t t);
void keyak_restart(Keyak * k);

void keyak_add_nonce(Keyak * k, uint8_t * nonce, uint32_t len);

void keyak_set_suv(Keyak * k, uint8_t * key, uint32_t klen);

void keyak_encrypt(Keyak * k, uint8_t * data, uint32_t datalen, 
                    uint8_t * metadata, uint32_t metalen);

void keyak_decrypt(Keyak * k, uint8_t * data, uint32_t datalen, 
                    uint8_t * metadata, uint32_t metalen, 
                    uint8_t * tag, uint32_t taglen);

#endif
