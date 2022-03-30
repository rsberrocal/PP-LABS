#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

char *trim(char *str) {
  char *end;

  // Trim leading space
  while(isspace((unsigned char)*str)) str++;

  if(*str == 0)  // All spaces?
    return str;

  // Trim trailing space
  end = str + strlen(str) - 1;
  while(end > str && isspace((unsigned char)*end)) end--;

  // Write new null terminator character
  end[1] = '\0';

  return str;
}

void readPPM(const char* fileName, unsigned int* width, unsigned int* height, 
             unsigned int* maximum, unsigned char** Pin) {
   FILE *fp;
   char *line = NULL;
   size_t len = 0;
   ssize_t read;
   char s[2] = " ";

   fp = fopen(fileName, "r");
   if (fp == NULL){
       printf("File not found\n");
   } else {

      /******* HEADER ********/
      read = getline(&line, &len, fp);
      if (strncmp(line, "P3", 2)) {
	  	   printf("Incorrect header found: %s\n", line);
      } else {
         read = getline(&line, &len, fp);
         char *p = strtok(line, s);
         *width = atoi(p);
         p = strtok (NULL, s);
         *height = atoi(p);
         read = getline(&line, &len, fp);
         *maximum = atoi(line);

         /******* DATA ********/
         int imageSizeInBytes = 3 * (*width) * (*height) * sizeof(unsigned char);
         *Pin = (unsigned char *)malloc(imageSizeInBytes);
         unsigned int i = 0;
         while ((read = getline(&line, &len, fp)) != -1) {
            line=trim(line);
            char *l = strtok (line, s);

            while (l != NULL)  {
               unsigned char* curPointer = *Pin + i++;
               *curPointer = (unsigned char) atoi(l);
               l = strtok (NULL, s);
            }
         }
         free(line);
      }
   }
	 fclose(fp);
}

void writePPM(const char* fileName, unsigned int width, unsigned int height, 
              unsigned int maximum, unsigned char* P) {
   unsigned int i, j, idx;
   FILE *fp = fopen(fileName, "w"); /* b - binary mode */
   (void) fprintf(fp, "P3\n%d %d\n%d\n", width, height, maximum);
   for (j = 0; j < height; ++j) {
      for (i = 0; i < width; ++i) {
         idx = width*j+i;
         (void) fprintf(fp, "%d %d %d ", P[idx*3] , P[idx*3+1], P[idx*3+2]);
      }
      (void) fprintf(fp, "\n");
   }
   (void) fclose(fp);
}

void writePGM(const char* fileName, unsigned int width, unsigned int height, 
              unsigned int maximum, unsigned char* P) {
   unsigned int i, j, idx;
   FILE *fp = fopen(fileName, "w"); /* b - binary mode */
   (void) fprintf(fp, "P2\n%d %d\n%d\n", width, height, maximum);
   for (j = 0; j < height; ++j) {
      for (i = 0; i < width; ++i) {
         idx = width*j+i;
         (void) fprintf(fp, "%d ", P[idx]);
      }
      (void) fprintf(fp, "\n");
   }
   (void) fclose(fp);
}

/*
int main(){
   char* inFileName = "lenaP3.ppm";
   char* outFileName = "lenaP3_out.ppm";
   unsigned int width = 0;
   unsigned int height = 0;
   unsigned int maximum = 0;
   unsigned char *P;
  
   readPPM(inFileName, &width, &height, &maximum, &P);
   writePPM(outFileName, width, height, maximum, P);

   printf("Width: %d\n", width);
   printf("Height: %d\n", height);
   printf("maximum: %d\n", maximum);
	 free(P);
   return 0;
}
*/
