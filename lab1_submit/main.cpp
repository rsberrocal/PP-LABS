#include <iostream>
#include <omp.h>
#include <chrono>

#define WIDTH 3840
#define HEIGHT 2160

#define EXPERIMENT_ITERATIONS 100

typedef unsigned char uchar;

struct _uchar3 {
uchar x;
uchar y;
uchar z;
}__attribute__((aligned (4)));

using uchar3 = _uchar3;

struct _uchar4 {
    uchar x;
    uchar y;
    uchar z;
    uchar w;
};

using uchar4 = _uchar4;

bool checkResults(uchar4* rgba, uchar3* grb, int size) {

    bool correct = true;

    for (int i=0; i < size; ++i) {        
        correct &= rgba[i].x == grb[i].y;
        correct &= rgba[i].y == grb[i].x;
        correct &= rgba[i].z == grb[i].z;
        correct &= rgba[i].w == 255;        
    }

    return correct;
}

// Function optimized with 2 loops paralleled by rows
void convertGRB2RGBA3_rows(uchar3* grb, uchar4* rgba, int width, int height) {
	#pragma omp parallel 
	{
	for (int x=0; x<height; ++x) {
		#pragma omp for
		for (int y=0; y<width; ++y) {	
			rgba[height * y + x].x = grb[height * y + x].y;
			rgba[height * y + x].y = grb[height * y + x].x;
			rgba[height * y + x].z = grb[height * y + x].z;
			rgba[height * y + x].w = 255;
		}
	}
	}              
}

// Function optimized with 2 loops paralleled by columns
void convertGRB2RGBA3_columns(uchar3* grb, uchar4* rgba, int width, int height) {
	for (int x=0; x<height; ++x) {
		#pragma omp parallel 
		{
		for (int y=0; y<width; ++y) {
			#pragma omp single
			{
			rgba[height * y + x].x = grb[height * y + x].y;
			rgba[height * y + x].y = grb[height * y + x].x;
			rgba[height * y + x].z = grb[height * y + x].z;
			rgba[height * y + x].w = 255;
			}
		}
		}
	}              
}

// Function optimized with 1 loop paralleled 
void convertGRB2RGBA3(uchar3* grb, uchar4* rgba, int width, int height) {
	#pragma omp parallel for
    for(int x = 0; x < width * height; x++){
        rgba[x].x = grb[x].y;
        rgba[x].y = grb[x].x;
        rgba[x].z = grb[x].z;
        rgba[x].w = 255;
    }    
}


// Function optimized by two loops
void convertGRB2RGBA2_(uchar3* grb, uchar4* rgba, int width, int height) {
	{
	for (int x=0; x<height; ++x) {
		for (int y=0; y<width; ++y) {	
			rgba[height * y + x].x = grb[height * y + x].y;
			rgba[height * y + x].y = grb[height * y + x].x;
			rgba[height * y + x].z = grb[height * y + x].z;
			rgba[height * y + x].w = 255;
		}
	}
	}              
}

// Function optimized by one loops
void convertGRB2RGBA2(uchar3* grb, uchar4* rgba, int width, int height) {
    for(int x = 0; x < width * height; x++){
        rgba[x].x = grb[x].y;
        rgba[x].y = grb[x].x;
        rgba[x].z = grb[x].z;
        rgba[x].w = 255;
    }                
}

void convertGRB2RGBA(uchar3* grb, uchar4* rgba, int width, int height) {
    for (int x=0; x<width; ++x) {
    	for (int y=0; y<height; ++y) {	
	    rgba[width * y + x].x = grb[width * y + x].y;
	    rgba[width * y + x].y = grb[width * y + x].x;
	    rgba[width * y + x].z = grb[width * y + x].z;
	    rgba[width * y + x].w = 255;
	}
    }
}

int main() {

    uchar3 *h_grb;
    uchar4 *h_rgba;

    int bar_widht = (HEIGHT/3) * WIDTH;

    // Alloc and generate GRB bars.
    h_grb = (uchar3*)malloc(sizeof(uchar3)*WIDTH*HEIGHT);
    for (int i=0; i < WIDTH * HEIGHT; ++i) {
        if (i < bar_widht) { h_grb[i] = { 255, 0, 0 }; }		//x 
        else if (i < bar_widht*2) { h_grb[i] = { 0, 255, 0 }; }	//y
        else { h_grb[i] = { 0, 0, 255 }; }				//z
    }

    // Alloc RGBA pointers
    h_rgba = (uchar4*)malloc(sizeof(uchar4)*WIDTH*HEIGHT);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<EXPERIMENT_ITERATIONS; ++i) {    
		convertGRB2RGBA3_rows(h_grb, h_rgba, WIDTH, HEIGHT);
		//convertGRB2RGBA3_columns(h_grb, h_rgba, WIDTH, HEIGHT);
		//convertGRB2RGBA3(h_grb, h_rgba, WIDTH, HEIGHT);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "convertGRB2RGBA time for " << EXPERIMENT_ITERATIONS \
    << " iterations = "<< duration << "us" << std::endl;

    bool ok = checkResults(h_rgba, h_grb, WIDTH*HEIGHT);

    if (ok) {
        std::cout << "Executed!! Results OK." << std::endl;
    } else {
        std::cout << "Executed!! Results NOT OK." << std::endl;
    }

    return 0;

}
