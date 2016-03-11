#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "keyak.h"

static void dump_hex(uint8_t * buf, int len)
{
    while(len--)
        printf("%x", *buf++);
    printf("\n");
}



int main(int argc, char * argv[])
{
    Keyak sendr;
    Keyak recvr;
    char * suv, * nonce;
    char pt[1024];
    int ptlen, suvlen, noncelen;

    if (argc == 3)
    {
        suv = argv[1];
        nonce = argv[2];
    }
    else if (argc == 2)
    {
        suv = argv[1];
        nonce = NULL;
    }
    else
    {
        fprintf(stderr, "usage: %s <key-ascii> [<nonce-ascii>]\n", argv[0]);
        exit(1);
    }
    suvlen = strlen(suv);
    noncelen = strlen(nonce);

    ptlen = read(STDIN_FILENO, pt, sizeof(pt));

    printf("plain text: \n");
    dump_hex(pt, ptlen);

    char metadata[] = "movie quote.";

    // lunar keyak
    keyak_init(&sendr,1600,12,256,128);
    keyak_init(&recvr,1600,12,256,128);

    keyak_set_suv(&sendr, suv, suvlen);
    keyak_set_suv(&recvr, suv, suvlen);
    keyak_add_nonce(&sendr, nonce, noncelen);
    keyak_add_nonce(&recvr, nonce, noncelen);

    keyak_encrypt(&sendr, pt, ptlen, metadata, sizeof(metadata));

    printf("cipher text: \n");
    dump_hex(sendr.O.buf, sendr.O.length);

    keyak_decrypt(&recvr, sendr.O.buf, sendr.O.length, 
                metadata, sizeof(metadata),
                sendr.T.buf, sendr.T.length);

    printf("plain text: \n");
    dump_hex(recvr.O.buf, recvr.O.length);


    printf("hello keyak\n");

    return 0;
}
