#ifndef _KEYAK_H_
#define _KEYAK_H_

#include <stdint.h>
#include "motorist.h"
#include "engine.h"
#include "piston.h"
#include "fleet.h"

typedef struct _Keyak
{
    Buffer T;
    Buffer SUV;
    Motorist motorist;
    Fleet * fleet;
} Keyak;


void keyak_init(Keyak* k, Fleet * fleet);
void keyak_restart(Keyak * k);

void keyak_add_nonce(Keyak * k, uint8_t * nonce, uint32_t len);

void keyak_set_suv(Keyak * k, uint8_t * key, uint32_t klen);

void keyak_encrypt(Keyak * k);

void keyak_decrypt(Keyak * k);

void keyak_destroy(Keyak * k);

#endif
