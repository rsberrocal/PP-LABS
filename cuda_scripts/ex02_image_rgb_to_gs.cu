#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "io_ppm.h"

#define BLOCK_SIZE 16

__global__ void RGBToGreyscaleKernel(unsigned char * Pout, unsigned char * Pin, 
                                     int width, int height) {
   int Col = //TODO: which column?;
   int Row = //TODO: which row?;

   if (// TODO: Is everything allowed?) {
      int greyOffset = Row*width + Col; // get 1D coordinate for the grayscale image

      unsigned char r = //TODO: pixel’s red value ;
      unsigned char g = //TODO: pixel’s green value ;
      unsigned char b = //TODO: pixel’s blue value ;

      // We multiply by floating point constants
      Pout[ //TODO: where? ] = (unsigned char)(0.21f*r + 0.72f*g + 0.07f*b);
   }
}

// Host code
int main() {
   // Host variables a & b
   unsigned char *h_P_in, *h_P_out, *d_P_in, *d_P_out;
   unsigned int width = 0;
   unsigned int height = 0;
   unsigned int maximum = 0;

   const char* inFileName = "lenaP3.ppm";
   const char* outFileName = "lenaP3_gs_out.pgm";

   readPPM(inFileName, &width, &height, &maximum, &h_P_in);

   int gsImageSizeInBytes = 3 * width * height * sizeof(unsigned char);

   h_P_out = (unsigned char *)malloc(gsImageSizeInBytes);

   CU_CHECK(
      cudaMalloc((void **)&d_P_in, gsImageSizeInBytes*3)
   );
   CU_CHECK(
      cudaMalloc((void **)&d_P_out, gsImageSizeInBytes)
   );

   CU_CHECK(
      cudaMemcpy(d_P_in, h_P_in, gsImageSizeInBytes*3, cudaMemcpyHostToDevice)
   );

   dim3 dimGrid(//TODO: define here the dimension of the grid [horizontal axis],
                //TODO: [vertical axis] );
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
   printf("%d,%d\n",height,width);
   printf("Grid (%d,%d,%d)\n",dimGrid.x,dimGrid.y,dimGrid.z);
   printf("Block (%d,%d,%d)\n",dimBlock.x,dimBlock.y,dimBlock.z);
   RGBToGreyscaleKernel<<<dimGrid, dimBlock>>>(d_P_out, d_P_in, width, height);
   CU_CHECK_LAST_ERROR;

   CU_CHECK(
      cudaMemcpy(h_P_out, d_P_out, gsImageSizeInBytes, cudaMemcpyDeviceToHost)
   );

   writePGM(outFileName, width, height, maximum, h_P_out);

   CU_CHECK(
      // Free device memory
      cudaFree(d_P_in)
   );
   CU_CHECK(
      // Free device memory
      cudaFree(d_P_out)
   );
   free(h_P_in);
   free(h_P_out);
   return 0;
}

