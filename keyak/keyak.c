
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "keyak.h"
#include "motorist.h"
#include "misc.h"


void keyak_init(Keyak* k)
{
    motorist_init(&k->motorist);
    buffer_init(&k->T,NULL,0);
    buffer_init(&k->SUV,NULL,0);
}

void keyak_restart(Keyak * k)
{
    motorist_restart(&k->motorist);
    k->T.offset = 0;
    k->T.length= 0;
    k->O.offset = 0;
    k->O.length= 0;
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


void keyak_encrypt(Keyak * k, uint8_t * data, uint32_t datalen, 
                    uint8_t * metadata, uint32_t metalen)
{

    Packet pkt;
    memset(&pkt, 0, sizeof(Packet));
    pkt.input = data;
    pkt.input_size = datalen;
    pkt.metadata = metadata;
    pkt.metadata_size = metalen;


    /*buffer_init(&k->I,data, datalen);*/
    buffer_init(&k->O,NULL, 0);
    /*buffer_init(&k->A,metadata, metalen);*/

    motorist_start_engine(&k->motorist, &k->SUV, 0, &k->T, 0, 0);

    motorist_wrap(&k->motorist, &pkt, &k->O, &k->T, 0, 0);
}

void keyak_decrypt(Keyak * k, uint8_t * data, uint32_t datalen, 
                    uint8_t * metadata, uint32_t metalen, 
                    uint8_t * tag, uint32_t taglen)
{
    Buffer tagbuf;

    Packet pkt;
    memset(&pkt, 0, sizeof(Packet));
    pkt.input = data;
    pkt.input_size = datalen;
    pkt.metadata = metadata;
    pkt.metadata_size = metalen;

    motorist_start_engine(&k->motorist, &k->SUV, 0, &k->T, 0, 0);

    buffer_init(&k->I,data, datalen);
    buffer_init(&k->O,NULL, 0);
    buffer_init(&k->A,metadata, metalen);
    buffer_init(&tagbuf, tag, taglen);
    motorist_wrap(&k->motorist,&pkt,&k->O, &tagbuf, 1, 0);

    if (k->motorist.phase == MotoristFailed)
    {
        fprintf(stderr,"authentication failed\n");
        exit(1);
        //fprintf(stderr, "but going to ignore it to test performance\n");
    }

}




