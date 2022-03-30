#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "utils_matrices.h"


#define TILE_WIDTH 2
__global__ void MatrixProductKernel(float* P, float* M, float* N, 
                                    int sidelength) {
   __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
   __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
   int bx = blockIdx.x; int by = blockIdx.y;
   int tx = threadIdx.x; int ty = threadIdx.y;
   // Identify the row and column of the P element to work on
   int Row = by * TILE_WIDTH + ty;
   int Col = bx * TILE_WIDTH + tx;
   float Pvalue = 0;
   // Loop over the M and N tiles required to compute P element
   for (int ph = 0; ph < sidelength/TILE_WIDTH; ++ph) {
      // Collaborative loading of M and N tiles into shared memory
      ds_M[ty][tx] = M[Row*sidelength + ph*TILE_WIDTH + tx];
      ds_N[ty][tx] = N[(ph*TILE_WIDTH + ty)*sidelength + Col];

      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; ++k) {
         Pvalue += ### YOUR CODE HERE ###;
      }

      __syncthreads();
   }
   P[Row*sidelength + Col] = Pvalue;
}


// Host code
int main() {
    srand(11);   // Random generator initialization

   // Host variables a & b
   float *h_M_in, *h_N_in, *h_P_out, *d_M_in, *d_N_in, *d_P_out;
	 unsigned int sq_sidelength = 6;

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

   dim3 dimGrid(ceil((float)sq_sidelength / TILE_WIDTH),
                ceil((float)sq_sidelength / TILE_WIDTH));
   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
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
