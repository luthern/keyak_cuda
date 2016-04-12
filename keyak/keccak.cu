/* Author: Noah Luther                                            */
/* Keccak-p permutation for Keyak authenticated cipher.           */
/* Based on implementation of Keccak-f by Gerhard Hoffman.        */
/**/

#define IS_THIS_WORKING
#ifdef IS_THIS_WORKING

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <errno.h>
#include <cuda.h>

static uint64_t *d_data;

#define ROUNDS        12
#define R64(a,b,c) (((a) << b) ^ ((a) >> c)) /* works on the GPU also for 
                                                b = 64 or c = 64 */
static const uint64_t round_const[5][ROUNDS] = {
    {0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
     0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
     0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
     0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL}};

/* Rho-Offsets. Note that for each entry pair their respective sum is 64.
   Only the first entry of each pair is a rho-offset. The second part is
   used in the R64 macros. */
static const uint8_t rho_offsets[25][2] = {
       /*y=0*/         /*y=1*/         /*y=2*/         /*y=3*/         /*y=4*/
/*x=0*/{ 0,64}, /*x=1*/{44,20}, /*x=2*/{43,21}, /*x=3*/{21,43}, /*x=4*/{14,50},
/*x=1*/{ 1,63}, /*x=2*/{ 6,58}, /*x=3*/{25,39}, /*x=4*/{ 8,56}, /*x=0*/{18,46},
/*x=2*/{62, 2}, /*x=3*/{55, 9}, /*x=4*/{39,25}, /*x=0*/{41,23}, /*x=1*/{ 2,62},
/*x=3*/{28,36}, /*x=4*/{20,44}, /*x=0*/{ 3,61}, /*x=1*/{45,19}, /*x=2*/{61, 3},
/*x=4*/{27,37}, /*x=0*/{36,28}, /*x=1*/{10,54}, /*x=2*/{15,49}, /*x=3*/{56, 8}};

static const uint8_t a_host[25] = {
    0,  6, 12, 18, 24,
    1,  7, 13, 19, 20,
    2,  8, 14, 15, 21,
    3,  9, 10, 16, 22,
    4,  5, 11, 17, 23};

static const uint8_t b_host[25] = {
    0,  1,  2,  3, 4,
    1,  2,  3,  4, 0,
    2,  3,  4,  0, 1,
    3,  4,  0,  1, 2,
    4,  0,  1,  2, 3};

static const uint8_t c_host[25][3] = {
    { 0, 1, 2}, { 1, 2, 3}, { 2, 3, 4}, { 3, 4, 0}, { 4, 0, 1},
    { 5, 6, 7}, { 6, 7, 8}, { 7, 8, 9}, { 8, 9, 5}, { 9, 5, 6},
    {10,11,12}, {11,12,13}, {12,13,14}, {13,14,10}, {14,10,11},
    {15,16,17}, {16,17,18}, {17,18,19}, {18,19,15}, {19,15,16},
    {20,21,22}, {21,22,23}, {22,23,24}, {23,24,20}, {24,20,21}};

static const uint8_t d_host[25] = {
          0,  1,  2,  3,  4,
         10, 11, 12, 13, 14,
         20, 21, 22, 23, 24,
          5,  6,  7,  8,  9,
         15, 16, 17, 18, 19};

__device__ __constant__ uint8_t a[25];
__device__ __constant__ uint8_t b[25];
__device__ __constant__ uint8_t c[25][3];
__device__ __constant__ uint8_t d[25];
__device__ __constant__ uint8_t ro[25][2];
__device__ __constant__ uint64_t rc[5][ROUNDS];

__device__
void keccak_p_kernel(uint64_t *data) {
    int const t = threadIdx.x;
    int const s = threadIdx.x%5;

    __shared__ uint64_t A[25];
    __shared__ uint64_t C[25];
    __shared__ uint64_t D[25];

    if (t < 25) {
        A[t] = data[t];

        for(int i=0;i<ROUNDS;++i) { 
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }
    data[t] = A[t];
    }
}

void gpu_init_keccak_tables()
{
    /* copy the tables from host to GPU */
    HANDLE_ERROR(cudaMemcpyToSymbol(a, a_host, sizeof(a_host)));
    HANDLE_ERROR(cudaMemcpyToSymbol(b, b_host, sizeof(b_host)));
    HANDLE_ERROR(cudaMemcpyToSymbol(c, c_host, sizeof(c_host)));
    HANDLE_ERROR(cudaMemcpyToSymbol(d, d_host, sizeof(d_host)));
    HANDLE_ERROR(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
    HANDLE_ERROR(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));

    /* allocate space for the state on GPU */
    HANDLE_ERROR(cudaMalloc((void **)&d_data, 200));
}

/* Modifies state with 12 rounds of Keccak.
   Uses the LFSR round constants for Keyak.
*/
/*void call_keccak_basic_kernel(uint64_t * state) {*/

    /*[> copy the data from the state to the GPU <]*/
    /*HANDLE_ERROR(cudaMemcpy(d_data, state, 200, cudaMemcpyHostToDevice));*/

    /*[> permute the state <]*/
    /*keccak_p_kernel<<<1,32>>>(d_data);*/

    /*[> fetch the generated data <]*/
    /*HANDLE_ERROR(cudaMemcpy(state, d_data, 200, cudaMemcpyDeviceToHost));*/
/*}*/

void cleanup_state()
{
    /* clean up the tables on the GPU */
    HANDLE_ERROR(cudaFree(d_data));
}

void _HANDLE_ERROR(cudaError_t e, int line)
{
    if (e != cudaSuccess)
    {
        printf("line: %d. error %s\n", line, cudaGetErrorString(e));
        exit(1);
    }
}
#else

#include "keccak.h"

#define     UNROLL_CHILOOP


#if defined(__GNUC__)
#define ALIGN __attribute__ ((aligned(32)))
#elif defined(_MSC_VER)
#define ALIGN __declspec(align(32))
#else
#define ALIGN
#endif

#if defined(_MSC_VER)
#define ROL64(a, offset) _rotl64(a, offset)
#elif defined(UseSHLD)
    #define ROL64(x,N) ({ \
    register UINT64 __out; \
    register UINT64 __in = x; \
    __asm__ ("shld %2,%0,%0" : "=r"(__out) : "0"(__in), "i"(N)); \
    __out; \
    })
#else
#define ROL64(a, offset) ((((UINT64)a) << offset) ^ (((UINT64)a) >> (64-offset)))
#endif

__host__ __device__ static tKeccakLane KeccakF1600_GetNextRoundConstant( UINT8 *LFSR )
{
    tSmallUInt i;
    tKeccakLane    roundConstant;
    tSmallUInt doXOR;
    tSmallUInt tempLSFR;

    roundConstant = 0;
    tempLSFR = *LFSR;
    for(i=1; i<128; i <<= 1)
    {
        doXOR = tempLSFR & 1;
        if ((tempLSFR & 0x80) != 0)
            // Primitive polynomial over GF(2): x^8+x^6+x^5+x^4+1
            tempLSFR = (tempLSFR << 1) ^ 0x71;
        else
            tempLSFR <<= 1;

        if ( doXOR != 0 )
            roundConstant ^= (tKeccakLane)1ULL << (i - 1);
    }
    *LFSR = (UINT8)tempLSFR;
    return ( roundConstant );
}

__host__ __device__ void KeccakP1600_StatePermute(void *argState, UINT8 rounds, UINT8 LFSRinitialState)
{
    const UINT8 KeccakF_RotationConstants[25] =
    {
        1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14, 27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
    };

    const UINT8 KeccakF_PiLane[25] =
    {
        10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4, 15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
    };

    //#define DIVISION_INSTRUCTION
#if    defined(DIVISION_INSTRUCTION)
#define    MOD5(argValue)    ((argValue) % 5)
#else
    const UINT8 KeccakF_Mod5[10] =
    {
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    };
#define    MOD5(argValue)    KeccakF_Mod5[argValue]
#endif


    tSmallUInt x, y, round;
    tKeccakLane        temp;
    tKeccakLane        BC[5];
    tKeccakLane     *state;
    UINT8           LFSRstate;

    state = (tKeccakLane*)argState;
    LFSRstate = LFSRinitialState;
    round = rounds;
    do
    {
        // Theta
        for ( x = 0; x < 5; ++x )
        {
            BC[x] = state[x] ^ state[5 + x] ^ state[10 + x] ^ state[15 + x] ^ state[20 + x];
        }
        for ( x = 0; x < 5; ++x )
        {
            temp = BC[MOD5(x+4)] ^ ROL64(BC[MOD5(x+1)], 1);
            for ( y = 0; y < 25; y += 5 )
            {
                state[y + x] ^= temp;
            }
        }

        // Rho Pi
        temp = state[1];
        for ( x = 0; x < 24; ++x )
        {
            BC[0] = state[KeccakF_PiLane[x]];
            state[KeccakF_PiLane[x]] = ROL64( temp, KeccakF_RotationConstants[x] );
            temp = BC[0];
        }

        //    Chi
        for ( y = 0; y < 25; y += 5 )
        {
#if defined(UNROLL_CHILOOP)
            BC[0] = state[y + 0];
            BC[1] = state[y + 1];
            BC[2] = state[y + 2];
            BC[3] = state[y + 3];
            BC[4] = state[y + 4];
#else
            for ( x = 0; x < 5; ++x )
            {
                BC[x] = state[y + x];
            }
#endif
            for ( x = 0; x < 5; ++x )
            {
                state[y + x] = BC[x] ^((~BC[MOD5(x+1)]) & BC[MOD5(x+2)]);
            }
        }

        //    Iota
        state[0] ^= KeccakF1600_GetNextRoundConstant(&LFSRstate);
    }
    while( --round != 0 );
}

void gpu_init_keccak_tables()
{
}

#endif

