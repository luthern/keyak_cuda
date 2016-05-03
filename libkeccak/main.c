/*
Implementation by the Keccak, Keyak and Ketje Teams, namely, Guido Bertoni,
Joan Daemen, Michaël Peeters, Gilles Van Assche and Ronny Van Keer, hereby
denoted as "the implementer".

For more information, feedback or questions, please refer to our websites:
http://keccak.noekeon.org/
http://keyak.noekeon.org/
http://ketje.noekeon.org/

To the extent possible under law, the implementer has waived all copyright
and related or neighboring rights to the source code in this file.
http://creativecommons.org/publicdomain/zero/1.0/
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "KeccakCodePackage.h"
#include "misc.h"
#include <openssl/bn.h>
#include <openssl/err.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

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



int main(int argc, char* argv[])
{
    //return process(argc, argv);

    unsigned char * key, * nonce = NULL, * key_hex, * nonce_hex = NULL;
    unsigned char * pt, * ot, * metadata = NULL, * metadata_hex = NULL;
    BIGNUM * key_bn = NULL, * nonce_bn = NULL;
    char * output, * inputname;
    FILE * outputf, * inputf;
    unsigned int ptlen, keylen, noncelen, mlen, readlen;
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


    keylen = hex2bin(&key,key_hex);
    noncelen = hex2bin(&nonce,nonce_hex);
    mlen = hex2bin(&metadata, metadata_hex);

    fseek(inputf, 0L, SEEK_END);
    size_t amt = ftell(inputf);
    fseek(inputf, 0L, SEEK_SET);

    pt = (unsigned char*)malloc(amt);
    ot = (unsigned char*)malloc(amt);
    ptlen = fread(pt, 1, amt, inputf);
    
    struct timer t;
    int i, j, k;

    for (j=10; j < 11; j++)
    {
        memset(&t, 0, sizeof(struct timer));
        timer_start(&t, "10000 sessions");

        for (i = 0; i < iterations; i++)
        {
            for(k=0; k< j; k++)
            {
                LunarKeyak_Instance *instance;
                unsigned char * tag;
                unsigned char * output;

                LunarKeyak_Initialize(instance, key, keylen, nonce, noncelen, 0, tag, 0, 0);
                LunarKeyak_Wrap(instance, pt, output, ptlen, metadata, mlen, tag, 0, 0);
            }
        }
        timer_end(&t);
    }



//testOneKeyak( const unsigned char * key, unsigned int keySizeInBytes, const unsigned char * nonce, unsigned int nonceSizeInBytes, const unsigned char * AD, size_t ADlen, const unsigned char * input, size_t dataSizeInBytes)

    return 0;
}
