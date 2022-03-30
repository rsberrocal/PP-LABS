#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "io_ppm.h"

#define BLOCK_SIZE 16
#define BLUR_SIZE 5

__global__ void blurringKernel(unsigned char * Pout, unsigned char * Pin, 
                               int width, int height) {
   int Col = // TODO: which column?
   int Row = // TODO: which row? 

   if (// TODO: Is everything allowed?) {
      float r_pixVal = 0, g_pixVal = 0, b_pixVal = 0, no_pixels = 0;

      for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow){
         for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol){
            // TODO: store the needed information
         }
      }
      // TODO: the next line might be repeated for each color channel
      Pout[// TODO: where?] = (unsigned char) (r_pixVal/no_pixels);
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
   const char* outFileName = "lenaP3_blur_out.ppm";

   readPPM(inFileName, &width, &height, &maximum, &h_P_in);

   int imageSizeInBytes = 3 * width * height * sizeof(unsigned char);

   h_P_out = (unsigned char *)malloc(imageSizeInBytes);

   CU_CHECK(
      cudaMalloc((void **)&d_P_in, imageSizeInBytes)
   );
   CU_CHECK(
      cudaMalloc((void **)&d_P_out, imageSizeInBytes)
   );

   CU_CHECK(
      cudaMemcpy(d_P_in, h_P_in, imageSizeInBytes, cudaMemcpyHostToDevice)
   );

   dim3 dimGrid(//TODO: define here the dimension of the grid [horizontal axis],
                //TODO: [vertical axis] );
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
   printf("%d,%d\n",height,width);
   printf("Grid (%d,%d,%d)\n",dimGrid.x,dimGrid.y,dimGrid.z);
   printf("Block (%d,%d,%d)\n",dimBlock.x,dimBlock.y,dimBlock.z);
   blurringKernel<<<dimGrid, dimBlock>>>(d_P_out, d_P_in, width, height);
   CU_CHECK_LAST_ERROR;

   CU_CHECK(
      cudaMemcpy(h_P_out, d_P_out, imageSizeInBytes, cudaMemcpyDeviceToHost)
   );

   writePPM(outFileName, width, height, maximum, h_P_out);

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

