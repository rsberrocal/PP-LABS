#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

// Device code or Kernel
__global__ void kernel_add(int a, int b, int* __restrict d_c) {
   *d_c = a * b;
}

// Host code
int main() {
   // Host variables a & b
   int a = 3, b = 5, h_c = 0;

   // Host variable that will store a Device pointer wich we can later on 
   // download to the Host.
   // As this variable will contain pointers that are only valid in
   // the Device (the GPU) it will be invalid to access them from
   // Host code. We only can use them in the right cuda API calls
   // or inside a cuda Kernel.
   // So in this part of the code you won't be able to do d_c[0], for instance
   int *d_c;

   // Size of the data contained in variables a, b and c.
   int dataSize = sizeof(int);

   // Reserve Device memory using the cuda API
   // cudaMalloc will place a Device pointer inside d_c.
   CU_CHECK(
      cudaMalloc((void **)&d_c, dataSize)
   );

   // Launch kernel_add() kernel on GPU
   // Notice that a and b are not pointers. Therefore the kernel call will
   // copy their values but the variables inside the kernel will not be the same.
   // If we modify a and b inside the kernel, it will not change a and b in this
   // Host code. This, indeed is the same behavior as any C/C++ function call.
   // In the case of d_c, it will copy the pointer contained in d_c, 
   // so we will be able to modify the contents of d_c from the kernel. But to read 
   // them from this Host code, we will have to do something else.
   kernel_add<<<1,1>>>(a, b, d_c);
   CU_CHECK_LAST_ERROR;

   CU_CHECK(
      // Copy result back to host
      cudaMemcpy(&h_c, d_c, dataSize, cudaMemcpyDeviceToHost)
   );

   printf("Result of multiplying %d * %d is %d\n",a,b,h_c);

   CU_CHECK(
      // Free device memory
      cudaFree(d_c)
   );
   return 0;
}

