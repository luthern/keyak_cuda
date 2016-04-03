#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "keyak.h"
#include "misc.h"

#define NUM_ITERATIONS 1
//#define NUM_ITERATIONS 100

// vector 1
const unsigned char key_v1[] = 
    "\x5a\x4b\x3c\x2d\x1e\x0f\x00\xf1\xe2\xd3\xc4\xb5\xa6\x97\x88\x79";
const unsigned char nonce_v1[] =
    "\x6b\x4c\x2d\x0e\xef\xd0\xb1\x92\x72\x53\x34\x15\xf6\xd7\xb8\x99";
    //"\x64\x4c\x2d\x0e\xef\xd0\xb1\x92\x72\x53\x34\x15\xf6\xd7\xb8\x99";
const unsigned char AD_v1[] =
    "\x32\xf3\xb4\x75\x35\xf6";
const unsigned char plaintext_v1[] = 
    "\xe4\x65\xe5\x66\xe6\x67\xe7";
const unsigned char ciphertext_v1[] =
    "\x20\xfe\xc6\x15\x45\x02\xc4\x77\x6b\x6a\x02\xba\xd7\xf9\xd3\x31\xc9\x6b\x62\x6c\x49\xda\xf2";
//


int main(int argc, char * argv[])
{
    Keyak sendr;
    Keyak recvr;
    unsigned char * suv;
    unsigned char * pt, * metadata;
    int ptlen, suvlen, noncelen, mlen;

    suv = (unsigned char * )key_v1;
    pt = (unsigned char * )plaintext_v1;
    metadata = (unsigned char * )AD_v1;

    uint8_t nonce[150];
    memset(nonce,0,sizeof(nonce));

    suvlen = (sizeof(key_v1)-1);
    noncelen = sizeof(nonce);
    ptlen = (sizeof(plaintext_v1)-1);
    mlen = (sizeof(AD_v1)-1);

    memmove(nonce, nonce_v1, sizeof(nonce_v1)-1);

    engine_precompute();
    // lunar keyak
    keyak_init(&sendr);
    keyak_init(&recvr);

    keyak_set_suv(&sendr, suv, suvlen);
    keyak_set_suv(&recvr, suv, suvlen);
    keyak_add_nonce(&sendr, nonce, noncelen);
    keyak_add_nonce(&recvr, nonce, noncelen);

    printf("encrypting %d bytes\n", ptlen);
    struct timer t, tinit;
    memset(&t, 0, sizeof(struct timer));
    int i;
    timer_start(&t, "10000 sessions");
    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        timer_start(&tinit,"keyak_initx2");

        keyak_restart(&sendr);
        keyak_restart(&recvr);

        timer_accum(&tinit);

        keyak_encrypt(&sendr, (uint8_t*)pt, ptlen, (uint8_t*)metadata, mlen);

        keyak_decrypt(&recvr, sendr.O.buf, sendr.O.length, 
                metadata, mlen,
                sendr.T.buf, sendr.T.length);
    }
    timer_end(&t);
    timer_end(&tinit);

    motorist_timers_end();
    
    printf("calculated ciphertext:\n");
    dump_hex2( sendr.O.buf , sendr.O.length );
    dump_hex( sendr.T.buf, sendr.T.length );
    printf("expected ciphertext:\n");
    dump_hex((uint8_t*)ciphertext_v1, sizeof(ciphertext_v1)-1);


    //int len = sendr.O.length;
    //printf("first %d of cipher: \n",len);
    //dump_hex( sendr.O.buf ,  len);

    printf("hello keyak\n");

    return 0;
}
