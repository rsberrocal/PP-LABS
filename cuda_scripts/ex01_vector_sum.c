#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 10000000

void vector_sum(unsigned char *out, unsigned char *in_a, unsigned char *in_b, int n) {
   for (int i = 0; i < n; i++){
      out[i] = in_a[i] + in_b[i];
   }
}

int main(){
   unsigned char *in_a, *in_b, *out; 
   int mem_size = sizeof(unsigned char)*N;
   // Allocate memory
   in_a = (unsigned char*)malloc(mem_size);
   in_b = (unsigned char*)malloc(mem_size);
   out  = (unsigned char*)malloc(mem_size);

   // Initialize arrays
   for (int i = 0; i < N; i++){
      in_a[i] = 1;
      in_b[i] = 2;
   }

   // Main function
   vector_sum(out, in_a, in_b, N);

   // Verification
   for (int i = 0; i < N; i++){
      assert(out[i] - in_a[i] - in_b[i]==0);
   }

   printf("out[0] = %d\n", out[0]);
   printf("PASSED\n");

   return 0;
}
