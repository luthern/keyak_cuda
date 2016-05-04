
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



static int fleet_exhausted(Fleet * f)
{
    
    Motorist * mptr;
    for(mptr = fleet_first(f); !fleet_end(f); mptr = fleet_next(f))
    {
        if (mptr->phase != MotoristDone)
        {
            return 0;
        }
    }
    return 1;
}

static void fleet_prof(Fleet * f, int * numReady, int * numRiding, int * numWrapped, int * numDone)
{
    Motorist * mptr;
    *numReady = 0;
    *numRiding = 0;
    *numWrapped = 0;
    *numDone = 0;
    int i=0;

    for(mptr = fleet_first(f); !fleet_end(f); mptr = fleet_next(f))
    {
        i++;
        switch(mptr->phase)
        {
            case MotoristReady:
                *numReady = *numReady + 1;
                break;
            case MotoristRiding:
                *numRiding = *numRiding + 1;
                break;
            case MotoristWrapped:
                *numWrapped = *numWrapped + 1;
                break;
            case MotoristFailed:
            case MotoristDone:
                *numDone = *numDone + 1;
                break;
            default:
                fprintf(stderr,"incorrect state for stream %d\n", i);
                break;
        }
    }

}


void keyak_encrypt(Keyak * k)
{
    int i;
    int ready, riding, wrapped, done;

    if (!fleet_size(k->fleet))
    {
        return;
    }

    do
    {
        /*fleet_prof(k->fleet, &ready, &riding, &wrapped, &done);*/

        /*// nothing is happening so start one*/
        /*if (fleet_size(k->fleet) == ready)*/
        /*{*/
            /*motorist_start_engine(fleet_first(k->fleet), &k->SUV, 0, fleet_first(k->fleet)->tag, 0, 0);*/
        /*}*/
        /*else if (ready == wrapped)*/
        /*{*/
            /*// time to copy*/
        /*}*/

        for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
        {
            switch(mptr->phase)
            {
                case MotoristReady:
                    /*printf("MotoristREady\n");*/
                    motorist_start_engine(mptr, &k->SUV, 0, mptr->tag, 0, 0);
                    break;
                case MotoristRiding:
                    /*printf("MotoristRiding\n");*/
                    motorist_wrap(mptr, 0);
                    break;
                case MotoristWrapped:
                    /*printf("MotoristWrapped\n");*/
                    motorist_authenticate(mptr, mptr->tag, 0, 0);
                    break;
                case MotoristFailed:
                    /*printf("MotoristFailed\n");*/
                    fprintf(stderr,"authentication failed for %d stream\n", i);
                    exit(1);
                    break;
                case MotoristDone:
                    /*printf("MotoristDone\n");*/
                    break;
                case MotoristWaiting:
                    /*printf("MotoristWaiting\n");*/
                    break;
                default:
                    fprintf(stderr,"incorrect state for stream %d\n", i);
                    break;

            }
            i++;
        }
    }
    while(!fleet_exhausted(k->fleet));

}

void keyak_decrypt(Keyak * k)
{
    Motorist * mptr;
    int i = 0;

    do
    {
        for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))
        {
            switch(mptr->phase)
            {
                case MotoristReady:
                    /*printf("MotoristREady\n");*/
                    motorist_start_engine(mptr, &k->SUV, 0, mptr->tag, 0, 0);
                    break;
                case MotoristRiding:
                    /*printf("MotoristRiding\n");*/
                    motorist_wrap(mptr, 1);
                    break;
                case MotoristWrapped:
                    /*printf("MotoristWrapped\n");*/
                    motorist_authenticate(mptr, mptr->tag, 1, 0);
                    break;
                case MotoristFailed:
                    /*printf("MotoristFailed\n");*/
                    fprintf(stderr,"authentication failed for %d stream\n", i);
                    exit(1);
                    break;
                case MotoristDone:
                    /*printf("MotoristDone\n");*/
                    break;
                case MotoristWaiting:
                    /*printf("MotoristWaiting\n");*/
                    break;
                default:
                    fprintf(stderr,"incorrect state for stream %d\n", i);
                    break;

            }
            i++;
        }
    }
    while(!fleet_exhausted(k->fleet));


    /*[>motorist_fuel(&k->motorist, data, datalen, metadata, metalen);<]*/


    /*for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))*/
    /*{*/
        /*while(motorist_wrap(mptr, 1) == MOTORIST_NOT_DONE)*/
        /*{}*/
    /*}*/
    /*for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))*/
    /*{*/
        /*motorist_authenticate(mptr, mptr->tag, 1, 0);*/
    /*}*/

    /*for(mptr = fleet_first(k->fleet); !fleet_end(k->fleet); mptr = fleet_next(k->fleet))*/
    /*{*/
        /*if (mptr->phase == MotoristFailed)*/
        /*{*/
            /*exit(1);*/
        /*}*/
    /*}*/

}




