#include <stdio.h>
#include "utils.h"

// Host code
int main() {

   int nDevs=0;
   CU_CHECK(
      cudaGetDeviceCount(&nDevs)
   );
   printf("No. devices: %d\n", nDevs);

   cudaDeviceProp prop;
   for(int i=0; i<nDevs; i++){ 
      CU_CHECK(
          cudaGetDeviceProperties(&prop, i)
      );
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

      printf("  No. SMs: %d\n", prop.multiProcessorCount);
      printf("  Max.no. Blocks per dimension (Grid): (%d,%d,%d)\n", 
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
      printf("  No. Threads per Block: %d\n", prop.maxThreadsPerBlock);
      printf("   with a maximum per dimension of (%d,%d,%d)\n", 
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
   }
   return 0;
}

