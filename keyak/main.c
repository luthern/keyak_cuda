#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <openssl/bn.h>
#include <openssl/err.h>

#include "keyak.h"
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
    unsigned char * pt, * metadata = NULL, * metadata_hex = NULL;
    BIGNUM * key_bn = NULL, * nonce_bn = NULL;
    char * output;
    FILE * outputf;
    int ptlen, keylen, noncelen, mlen, readlen;
    int iterations = 1;

    if (argc < 3 || argc > 9)
    {
        fprintf(stderr, "usage: %s <key-hex> <output-file> [-n <nonce-hex>] [-m <metadata-hex>] [-i <iterations>]\n", argv[0]);
        exit(1);
    }

    key_hex = (unsigned char *)argv[1];
    output = argv[2];
    outputf = fopen(output,"w+");
    if (outputf == NULL)
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

    keyak_init(&sendr);
    keyak_init(&recvr);

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

    pt = (unsigned char*)malloc(5000);
    ptlen = read(STDIN_FILENO, pt, 5000);
    if (ptlen <= 0)
    {
        perror("read");
        goto done;
    }
    if (ptlen == 5000)
    {
        fprintf(stderr,"cannot encrypt more than 5000 bytes\n");
        goto done;
    }

    struct timer t, tinit;
    memset(&t, 0, sizeof(struct timer));
    int i;
    timer_start(&t, "10000 sessions");
    for (i = 0; i < iterations; i++)
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
    
    if (write(fileno(outputf),sendr.O.buf,sendr.O.length) == -1)
    {
        perror("write");
        goto done;
    }
    if (write(fileno(outputf),sendr.T.buf,sendr.T.length) == -1)
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

    //engine_destroy(k);

    return 0;
}
