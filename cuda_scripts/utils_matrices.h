#include <stdio.h>

void generateMatrix(float** mat, int width, int height){
   int S = width*height;
   *mat = (float *)malloc(S * sizeof(float));

   for (int i=0;i<S;i++) {
      (*mat)[i] = rand() % 9 + 1;
   }
}
void printMatrix(float* mat, int width, int height) {
   int it = 0;
   for (int i=0;i<height;i++) {
      for (int j=0;j<width;j++) {
         printf("%.0f ", mat[it++]);
      }
      printf("\n");
   }
}
