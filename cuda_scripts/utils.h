#include <stdio.h>

void cuCheck(cudaError_t err) {
   if(err!=cudaSuccess) {
      printf("Cuda error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err)); 
      exit(EXIT_FAILURE);
  }
}
#define CU_CHECK(a) cuCheck(a)
#define CU_CHECK_LAST_ERROR cuCheck(cudaGetLastError())

