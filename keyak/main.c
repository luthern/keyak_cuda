#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <openssl/bn.h>
#include <openssl/err.h>

#include "keyak.h"
#include "fleet.h"
#include "misc.h"

static void openssl_die()
{
    fprintf(stderr,"error: %s\n",
            ERR_error_string(ERR_get_error(),NULL) );
    exit(2);
}

static unsigned int hex2bin(unsigned char ** bin, const unsigned char * hex)
{
    int len;
    BIGNUM * bn = NULL;
    if(BN_hex2bn(&bn, (char*)hex) == 0)
    {   openssl_die();    }

    len = BN_num_bytes(bn);
    *bin = (unsigned char *)malloc(len);

    if(BN_bn2bin(bn, *bin) == 0)
    {   openssl_die();  }
    return len;
}

int main(int argc, char * argv[])
{
    Keyak sendr;
    Keyak recvr;
    unsigned char * key, * nonce = NULL, * key_hex, * nonce_hex = NULL;
    unsigned char * pt, * ot, * metadata = NULL, * metadata_hex = NULL;
    BIGNUM * key_bn = NULL, * nonce_bn = NULL;
    char * output, * inputname;
    FILE * outputf, * inputf;
    int ptlen, keylen, noncelen, mlen, readlen;
    int iterations = 1;

    if (argc < 4 || argc > 10)
    {
        fprintf(stderr, "usage: %s <key-hex> <input-file> <output-file> [-n <nonce-hex>] [-m <metadata-hex>] [-i <iterations>]\n", argv[0]);
        exit(1);
    }

    key_hex = (unsigned char *)argv[1];
    output = argv[3];
    inputname = argv[2];
    outputf = fopen(output,"w+");

    if (outputf == NULL)
    {
        perror("fopen");
        exit(1);
    }
    inputf = fopen(inputname,"r");
    if (inputf == NULL)
    {
        perror("fopen");
        exit(1);
    }


    ERR_load_crypto_strings();

    int opt;
    while ((opt = getopt (argc, argv, "n:m:i:")) != -1)
    {
        switch (opt)
        {
            case 'n':
                nonce_hex = (unsigned char *)optarg;
                break;
            case 'm':
                metadata_hex = (unsigned char *)optarg;
                break;
            case 'i':
                iterations = atoi(optarg);
                break;
            default:
                fprintf(stderr,"unrecognized argument -%c", (char)opt);
                exit(1);
                break;
        }
    }

    engine_precompute();
    Fleet * frecv = fleet_new(50,50);
    Fleet * fsend = fleet_new(50,50);

    keyak_init(&sendr, fsend);
    keyak_init(&recvr, frecv);

    keylen = hex2bin(&key,key_hex);

    keyak_set_suv(&sendr, key, keylen);
    keyak_set_suv(&recvr, key, keylen);

    if (nonce_hex != NULL)
    {
        noncelen = hex2bin(&nonce,nonce_hex);

        keyak_add_nonce(&sendr, nonce, noncelen);
        keyak_add_nonce(&recvr, nonce, noncelen);
    }

    if (metadata_hex != NULL)
    {
        mlen = hex2bin(&metadata, metadata_hex);
    }

    fseek(inputf, 0L, SEEK_END);
    size_t amt = ftell(inputf);
    fseek(inputf, 0L, SEEK_SET);

    pt = (unsigned char*)malloc(amt);
    ot = (unsigned char*)malloc(amt);
    ptlen = fread(pt, 1, amt, inputf);

    if (pt == NULL || ot == NULL)
    {
        perror("malloc");
    }

    printf("processing %d bytes\n",ptlen);

    if (ptlen <= 0 || ptlen != amt)
    {
        perror("fread");
        goto done;
    }


    struct timer t, tinit;
    int i,j,k;



    for (j=10; j < 11; j++)
    {

        memset(&t, 0, sizeof(struct timer));
        timer_start(&t, "10000 sessions");

        printf("STREAMS: %d\n", j);
        for (i = 0; i < iterations; i++)
        {
            /*timer_start(&tinit,"keyak_initx2");*/

            keyak_restart(&sendr);
            keyak_restart(&recvr);

            /*timer_accum(&tinit);*/

            for(k=0; k< j; k++)
            {
                fleet_add_stream(fsend, pt,ptlen,metadata,mlen,ot,ptlen, NULL);
            }

            keyak_encrypt(&sendr);

            /*fleet_add_stream(frecv, ot,ptlen,metadata,mlen,pt,ptlen, fleet_first(fsend)->tag);*/

            /*keyak_decrypt(&recvr);*/
        }

        timer_end(&t);
        /*timer_end(&tinit);*/
        engine_sync();


    }
    motorist_timers_end();
    
    if (write(fileno(outputf),fleet_first(fsend)->output,ptlen) == -1)
    {
        perror("write");
        goto done;
    }
    if (write(fileno(outputf),fleet_first(fsend)->tag,KEYAK_TAG_SIZE/8) == -1)
    {
        perror("write");
        goto done;
    }
    fflush(outputf);

done:
    keyak_destroy(&sendr);
    keyak_destroy(&recvr);
    free(pt);
    if (metadata != NULL) 
    {
        free(metadata);
    }
   
    if (nonce != NULL) 
    {
        free(nonce);
    }
    free(key);
    fclose(outputf);
    ERR_free_strings();

    fleet_destroy(frecv);
    fleet_destroy(fsend);

    //engine_destroy(k);

    return 0;
}

