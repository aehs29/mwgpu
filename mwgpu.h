/* General Includes */
#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <time.h>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

// OpenGl Includes
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

/* CUDA Includes */
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_gl_interop.h>

/* GLM */
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader_utils.h"

// Serial Displacement
#include "displacement.cpp"

/* Loading Files */
#include "file_loading.cpp"



// Functions
void ResizeFunction(int Width, int Height);
void IdleFunction(void);
void RenderFunction(void);
void CreateVBO_CUDA(void);
void processSpecialKeys(int key, int x, int y);
void releaseKey(int key, int x, int y) ;
void mouseMove(int x, int y);
void mouseButton(int button, int state, int x, int y) ;
void change_force(float *F, float *Fo, nodes_struct ns, float force, int force_axis);
void Cleanup(void);
void DestroyShaders(void);
void DestroyVBO(void);
void GLM_MVP(GLuint pId);


// Functions to call CUDA (compiled with nvcc)
extern "C" void map_Texture(void *cuda_data, size_t size,cudaGraphicsResource *resource);
extern "C" bool displacement (float *h_qo, float *h_qdo, float *h_Fo, float h_h, unsigned int eigencount, unsigned int node_count, unsigned int node_dimensions, float *simtime, const int block_size, float * buffer, int *fixed_nodes, unsigned int fixed_nodes_count, unsigned int maxThreadsBlock);
extern "C" bool free_GPUnodes();
extern "C" void allocate_GPUmem(float *nodes, float *h_alphaI, float *h_alpha, float *h_beta, float *h_gamma, float *h_eigenVecs, float *h_Psy, int node_count, int node_dimensions, int fixed_nodes_count,  int eigencount);


/* Performance Measurement */
timespec diff_time(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		/* Time in secs not needed */
 		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}
