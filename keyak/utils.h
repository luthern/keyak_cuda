#ifndef _UTILS_H_
#define _UTILS_H_

#include <cuda.h>

/*  misc/utils for gpu things
 * */

#define HANDLE_ERROR(e)  _HANDLE_ERROR(e, __FILE__,__LINE__)

void _HANDLE_ERROR(cudaError_t e,const char * file,  int line);

#endif
