#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "utils_matrices.h"

#define BLOCK_SIZE 2

__global__ void MatrixProductKernel(float* P, float* M, float* N, int sidelength) {
   int Col = threadIdx.x + blockIdx.x * blockDim.x;
   int Row = threadIdx.y + blockIdx.y * blockDim.y;

   if (Col < sidelength && Row < sidelength) {
      float Pvalue = 0;
      float mValue = 0;
      float nValue = 0;
      // each thread computes one element of the block sub-matrix
      for (int k = 0; k < sidelength; ++k) {
         mValue = M[Row * k + sidelength];
         nValue = N[Col * k + sidelength];
         Pvalue += mValue * nValue;
      }
      P[Row*sidelength+Col] = Pvalue;
   }
}

// Host code
int main() {
    srand(11);   // Random generator initialization

   // Host variables a & b
   float *h_M_in, *h_N_in, *h_P_out, *d_M_in, *d_N_in, *d_P_out;
   unsigned int sq_sidelength = 4;

   generateMatrix(&h_M_in, sq_sidelength, sq_sidelength);
   generateMatrix(&h_N_in, sq_sidelength, sq_sidelength);

   printf("M\n");
   printMatrix(h_M_in, sq_sidelength, sq_sidelength);
   printf("N\n");
   printMatrix(h_N_in, sq_sidelength, sq_sidelength);

   int mSizeInBytes = sq_sidelength * sq_sidelength * sizeof(float);

   h_P_out = (float *)malloc(mSizeInBytes);

   CU_CHECK(
      cudaMalloc((void **)&d_M_in, mSizeInBytes)
   );
   CU_CHECK(
      cudaMalloc((void **)&d_N_in, mSizeInBytes)
   );
   CU_CHECK(
      cudaMalloc((void **)&d_P_out, mSizeInBytes)
   );

   CU_CHECK(
      cudaMemcpy(d_M_in, h_M_in, mSizeInBytes, cudaMemcpyHostToDevice)
   );
   CU_CHECK(
      cudaMemcpy(d_N_in, h_N_in, mSizeInBytes, cudaMemcpyHostToDevice)
   );

   dim3 dimGrid(ceil((float)sq_sidelength / BLOCK_SIZE),
                ceil((float)sq_sidelength / BLOCK_SIZE));
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
   printf("Grid (%d,%d,%d)\n",dimGrid.x,dimGrid.y,dimGrid.z);
   printf("Block (%d,%d,%d)\n",dimBlock.x,dimBlock.y,dimBlock.z);
   MatrixProductKernel<<<dimGrid, dimBlock>>>(d_P_out, d_M_in, d_N_in,  
                                              sq_sidelength);
   CU_CHECK_LAST_ERROR;

   CU_CHECK(
      cudaMemcpy(h_P_out, d_P_out, mSizeInBytes, cudaMemcpyDeviceToHost)
   );

   printf("P (=MxN)\n");
   printMatrix(h_P_out, sq_sidelength, sq_sidelength);


   CU_CHECK(
      // Free device memory
      cudaFree(d_M_in)
   );
   CU_CHECK(
      // Free device memory
      cudaFree(d_N_in)
   );
   CU_CHECK(
      // Free device memory
      cudaFree(d_P_out)
   );
   free(h_M_in);
   free(h_N_in);
   free(h_P_out);
   return 0;
}
