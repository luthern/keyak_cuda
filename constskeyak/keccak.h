#ifndef _KECCAK_H_
#define _KECCAK_H_

typedef unsigned char UINT8;
typedef unsigned long long int UINT64;
typedef unsigned int tSmallUInt; /*INFO It could be more optimized to use "unsigned char" on an 8-bit CPU    */
typedef UINT64 tKeccakLane;


void KeccakP1600_StatePermute(void *argState, UINT8 rounds, UINT8 LFSRinitialState);

#define PERMUTE(state)      KeccakP1600_StatePermute(state, 12, 0xd5);

#endif
