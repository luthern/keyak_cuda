#ifndef _DEFS_H_
#define _DEFS_H_
/*
 *                    width   nr   capacity   tag size
    keyak_init(&recvr,1600,   12,  256,       128);
    m->Rs = W * ((KEYAK_F_WIDTH - MAX(32, c))/W) >> 3;
    m->Ra = W * ((KEYAK_F_WIDTH - 32)/W) >> 3;
    
    m->cprime = W*((c+W-1)/W);

    p->EOM = Ra;
    p->CryptEnd = Ra + 1;
    p->InjectStart = Ra + 2;
    p->InjectEnd = Ra + 3;
 
*/

#define MAX(a,b)        ((a) > (b) ? (a) : (b))
#define MIN(a,b)        ((a) < (b) ? (a) : (b))
#define CEIL(x,y)       (((x) + (y) - 1) / (y))

#define KEYAK_F_WIDTH           1600
#define KEYAK_NUM_PISTONS       8
#define KEYAK_NUM_ROUNDS        12
#define KEYAK_CAPACITY          256
#define KEYAK_TAG_SIZE          128
#define KEYAK_STATE_SIZE        (( KEYAK_F_WIDTH + 7 )/8)
#define KEYAK_WORD_SIZE         MAX(KEYAK_F_WIDTH/25,8)
#define KEYAK_CPRIME            (KEYAK_WORD_SIZE*((KEYAK_CAPACITY+KEYAK_WORD_SIZE-1)/KEYAK_WORD_SIZE))

#define PISTON_RS               (KEYAK_WORD_SIZE * ((KEYAK_F_WIDTH - MAX(32, KEYAK_CAPACITY))/KEYAK_WORD_SIZE) / 8)
#define PISTON_RA               (KEYAK_WORD_SIZE * ((KEYAK_F_WIDTH - 32)/KEYAK_WORD_SIZE) / 8)

#define PISTON_EOM              (PISTON_RA)
#define PISTON_CRYPT_END        (PISTON_RA+1)
#define PISTON_INJECT_START     (PISTON_RA+2)
#define PISTON_INJECT_END       (PISTON_RA+3)


#define KEYAK_BUFFER_SIZE       5000
#define KEYAK_GPU_BUF_SLOTS     32

#endif
