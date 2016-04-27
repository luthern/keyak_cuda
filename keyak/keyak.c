
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "keyak.h"
#include "motorist.h"
#include "fleet.h"
#include "misc.h"


void keyak_init(Keyak* k, Fleet * f)
{
    motorist_init(&k->motorist);
    buffer_init(&k->T,NULL,0);
    buffer_init(&k->SUV,NULL,0);
    k->fleet = f;
}

void keyak_destroy(Keyak * k)
{
    motorist_destroy(&k->motorist);
}

void keyak_restart(Keyak * k)
{
    Motorist * mptr;

    for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
    {
        motorist_restart(mptr);
    }
    k->fleet->streams = 0;


    k->T.offset = 0;
    k->T.length= 0;
    k->SUV.offset = 0;
    k->SUV.length;
}


void keyak_set_suv(Keyak * k, uint8_t * key, uint32_t klen)
{
    int i;
    uint32_t lk = KEYAK_WORD_SIZE/8 * CEIL(KEYAK_CAPACITY + 9, KEYAK_WORD_SIZE);
    assert(klen <= lk - 2);
    buffer_put(&k->SUV, lk);

    for(i = 0; i < klen; i++)
    {
        buffer_put(&k->SUV, key[i]);
    }
    buffer_put(&k->SUV, 1);
    while(k->SUV.length < lk)
    {
        buffer_put(&k->SUV, 0);
    }
    assert(k->SUV.buf[0] == k->SUV.length);
}

void keyak_add_nonce(Keyak * k, uint8_t * nonce, uint32_t len)
{
    int i;
    for (i = 0; i < len; i++)
    {
        buffer_put(&k->SUV, nonce[i]);
    }
}



void keyak_encrypt(Keyak * k)
{
    Motorist * mptr;

    for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
    {
        /*printf( "motorist ready? %d\n",mptr->phase == MotoristReady);*/
        motorist_start_engine(mptr, &k->SUV, 0, mptr->tag, 0, 0);
    }

    /*for(mptr = fleet_first(); !fleet_end(); mptr = fleet_next())*/
    /*{*/
        /*motorist_fuel(mptr, data, datalen, metadata, metalen);*/
    /*}*/

    for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
    {
        while(motorist_wrap(mptr, 0) == MOTORIST_NOT_DONE)
        {}
    }
    for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
    {
        motorist_authenticate(mptr, mptr->tag, 0, 0);
    }
}

void keyak_decrypt(Keyak * k)
{
    Motorist * mptr;

    for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
    {
        /*printf( "motorist ready? %d\n",mptr->phase == MotoristReady);*/
        motorist_start_engine(mptr, &k->SUV, 0, mptr->tag, 0, 0);
    }


    /*motorist_fuel(&k->motorist, data, datalen, metadata, metalen);*/


    for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
    {
        while(motorist_wrap(mptr, 1) == MOTORIST_NOT_DONE)
        {}
    }
    for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
    {
        motorist_authenticate(mptr, mptr->tag, 1, 0);
    }

    int i = 0;
    for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
    {
        if (mptr->phase == MotoristFailed)
        {
            fprintf(stderr,"authentication failed for %d stream\n", i);
            exit(1);
        }
        i++;
    }

}




