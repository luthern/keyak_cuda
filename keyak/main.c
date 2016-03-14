#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "keyak.h"
#include "misc.h"
#include "add.h"

static void dump_hex(uint8_t * buf, int len)
{
    while(len--)
        printf("%02hhx", *buf++);
    printf("\n");
}



int main(int argc, char * argv[])
{
    Keyak sendr;
    Keyak recvr;
    unsigned char * suv, * nonce;
    char pt[5024];
    int ptlen, suvlen, noncelen;

    do_sum_adds();

    if (argc == 3)
    {
        suv = (unsigned char *) argv[1];
        nonce = (unsigned char *)argv[2];
    }
    else if (argc == 2)
    {
        suv = (unsigned char *)argv[1];
        nonce = NULL;
    }
    else
    {
        fprintf(stderr, "usage: %s <key-ascii> [<nonce-ascii>]\n", argv[0]);
        exit(1);
    }
    suvlen = strlen((char*)suv);
    noncelen = strlen((char*)nonce);

    ptlen = read(STDIN_FILENO, pt, sizeof(pt));

    //printf("plain text: \n");
    //dump_hex(pt, ptlen);

    unsigned char metadata[] = "movie quote.";
    
    engine_precompute();
    // lunar keyak
    keyak_init(&sendr,1600,12,256,128);
    keyak_init(&recvr,1600,12,256,128);

    keyak_set_suv(&sendr, suv, suvlen);
    keyak_set_suv(&recvr, suv, suvlen);
    keyak_add_nonce(&sendr, nonce, noncelen);
    keyak_add_nonce(&recvr, nonce, noncelen);

    printf("encrypting %d bytes\n", ptlen);
    struct timer t, tinit;
    memset(&t, 0, sizeof(struct timer));
    int i;
    timer_start(&t, "10000 sessions");
    for (i=0; i< 20000; i++)
    {
        timer_start(&tinit,"keyak_initx2");

        keyak_restart(&sendr);
        keyak_restart(&recvr);

        timer_accum(&tinit);

        keyak_encrypt(&sendr, (uint8_t*)pt, ptlen, (uint8_t*)metadata, sizeof(metadata));

        keyak_decrypt(&recvr, sendr.O.buf, sendr.O.length, 
                metadata, sizeof(metadata),
                sendr.T.buf, sendr.T.length);
    }
    timer_end(&t);
    timer_end(&tinit);

    motorist_timers_end();
    dump_hex( sendr.T.buf, sendr.T.length );
    dump_hex( sendr.O.buf + sendr.O.length - 50,50 );

    printf("hello keyak\n");

    return 0;
}
