#ifndef CUDA_KECCAK_BASIC_CUH_INCLUDED
#define CUDA_KECCAK_BASIC_CUH_INCLUDED

#ifndef HANDLE_ERROR
#define HANDLE_ERROR(e) _HANDLE_ERROR(e, __LINE__)
#endif

#include <cuda.h>

void _HANDLE_ERROR(cudaError_t e, int line);

#include <inttypes.h>

void gpu_init_keccak_tables();
void cleanup_state();
void call_keccak_basic_kernel(uint64_t * state);

#define PERMUTE(state)    call_keccak_basic_kernel(state)

#endif /* CUDA_KECCAK_BASIC_CUH_INCLUDED */

