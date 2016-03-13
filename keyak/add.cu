#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>


#define N 10

void add( int *a, int *b, int *c ) {
    int tid = 0; // this is CPU zero, so we start at zero
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += 1; // we have one CPU, so we increment by one
    }
}

__global__ void add_gpu( int *a, int *b, int *c ) {
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

#define HANDLE_ERROR(e)  _HANDLE_ERROR(e, __LINE__)

void _HANDLE_ERROR(cudaError_t e, int line)
{
    if (e != cudaSuccess)
    {
        printf("line: %d. error %s\n", line, cudaGetErrorString(e));
        exit (1);
    }
}

int do_sum_adds()
{
    int a[N], b[N], c[N];
    int * dev_a, * dev_b, * dev_c;
    int i;

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

    for (i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                cudaMemcpyHostToDevice ) );

    //add(a,b,c);
    add_gpu<<<N,1>>>( dev_a, dev_b, dev_c );

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
           cudaMemcpyDeviceToHost) );

    // display the results
    for (i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
    
    return 0;
}
