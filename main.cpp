#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <time.h>


// OpenGl Includes
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

/* CUDA Includes */
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_gl_interop.h>



#include "displacement.cpp"

using namespace std;


int sizeBuffer;


float *nodes;
GLshort *elem;
float *eigenVals;
float *eigenVecs;
int *fixed_nodes;
float *d_nodes;


// Declare Coefficients 
// Square matrices dimensions: #eigenvals
float *d,*alpha, *alphaI,*beta,*gama,*M,*C;
float *F, *Fo, *q, *qo, *qd, *qdo, *u, *R, *Ro;
float h;



int elem_count, elem_nodes;
int node_count, node_dimensions;
int eigencount;
unsigned int fixed_nodes_count;


bool render=true;
float red=1.0f, blue=0.0f, green=0.0f;

// angle for rotating triangle
float angle = 0.0f;
float deltaAngle = 0.0f;
int xOrigin = -1;

float angleX = 0.0f;
float deltaAngleX = 0.0f;
int yOrigin=-1;

float scale=1.0;
cudaGraphicsResource *resources[1];
//cudaGraphicsResource *original_nodes[1];


//extern "C" bool MatrixMult (float *h_A, float *h_B, float *h_C , int RowsA, int ColsA, int RowsB, int ColsB, const int block_size);

//extern "C" bool runTest2(const int argc, const char **argv, int i);
	bool bTestResult;
extern "C" bool runTest(const int argc, const char **argv, float * buffer, float *u,  int node_count, float scale);
extern "C" void map_texture(void *cuda_data, size_t size,cudaGraphicsResource *resource);

extern "C" bool displacement (float *h_q, float *h_qo, float *h_qd, float *h_qdo, float *h_F, float *h_Fo, float *h_Ro, float *h_alpha, float * h_alphaI, float *h_beta, float *h_gama, float *h_eigenVecs, float h_h, float *h_u, unsigned int eigencount, unsigned int node_count, unsigned int node_dimensions, const int block_size, float *buffer, float *h_nodes, int *fixed_nodes, unsigned int fixed_nodes_count, float scale);

extern "C" void* allocate_GPUnodes(float *d_nodes, float *nodes, unsigned int node_count, unsigned int node_dimensions);
extern "C" bool free_GPUnodes(float *d_nodes);




void DestroyShaders(void);
void DestroyVBO(void);



size_t BufferSize,VertexSize;
int CurrentWidth = 600,
	CurrentHeight = 480,
	WindowHandle = 0;

GLuint
	VertexShaderId,
	FragmentShaderId,
	ProgramId,
	VaoId,
	BufferId,
	IndexBufferId[2],
	ActiveIndexBuffer = 0;

const GLchar* VertexShader =
{
	"#version 400\n"\

	"layout(location=0) in vec3 in_Position;\n"\
	"layout(location=0) in vec3 in_Color;\n"\
	"out vec4 ex_Color;\n"\

	"void main(void)\n"\
	"{\n"\
	"	gl_Position = vec4(in_Position,1.0);\n"\
	"	ex_Color = vec4(in_Color,1.0);\n"\
	"}\n"
};

const GLchar* FragmentShader =
{
	"#version 400\n"\

	"in vec4 ex_Color;\n"\
	"out vec4 out_Color;\n"\

	"void main(void)\n"\
	"{\n"\
	"	out_Color = ex_Color;\n"\
	"}\n"
};


size_t siz;
float *cuda_dat = NULL;


/*
void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;
	float ratio =  w * 1.0 / h;

        // Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

        // Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45.0f, ratio, 0.1f, 100.0f);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);

	// Draw on cuda?
	glEnable(GL_DEPTH_TEST);


}

*/


void ResizeFunction(int Width, int Height)
{
if (Height == 0)
		Height = 1;
	float ratio =  Width * 1.0 / Height;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	CurrentWidth = Width;
	CurrentHeight = Height;
	glViewport(0, 0, CurrentWidth, CurrentHeight);
//	gluPerspective(45.0f, ratio, 0.1f, 100.0f); // ULTRA DANGER
	glMatrixMode(GL_MODELVIEW);
}

void IdleFunction(void)
{
	glutPostRedisplay();
}

void RenderFunction(void)
{

//	if (deltaMove)
//		computePos(deltaMove);
	//++FrameCount;
	angle+=deltaAngle;
	angleX+=deltaAngleX;
	
	//Modify();

//void * ptr;
//cudaGLMapBufferObject(&ptr,BufferId);

//cudaStream_t cuda_stream;
 
//Create CUDA stream
//cudaStreamCreate(&cuda_stream);

//let's have a look at the buffer
    // glBindBuffer(GL_ARRAY_BUFFER, BufferId);
    // float* mappedBuffer = (float *) glMapBuffer(GL_ARRAY_BUFFER,GL_READ_ONLY);
    // printf("\tAfter Kernel Execution:\n");
	// 	   for(int h=0;h<node_count*node_dimensions;h++){
	// 		   printf("Index: %d, Value: %f\n",h,mappedBuffer[h]);
	// 	   }
    // glUnmapBuffer(GL_ARRAY_BUFFER);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);

   










//Map the graphics resource to the CUDA stream

	if (cudaGraphicsMapResources(1, resources,0) != cudaSuccess)
        printf("Resource mapping failed...\n");
	// if (cudaGraphicsMapResources(1, original_nodes,0) != cudaSuccess)
    //     printf("Resource mapping failed...\n");
	if(cudaGraphicsResourceGetMappedPointer((void**)&cuda_dat, &siz, *resources) !=cudaSuccess)
	    printf("Resource pointer mapping failed...\n");
	// if(cudaGraphicsResourceGetMappedPointer((void**)&cuda_dat_original, &siz, *original_nodes) !=cudaSuccess)
	//     printf("Resource pointer mapping failed...\n");



	// Get displacement

/* Parallel Code*/


	const int block_size=16;
	bool disp ;
//	long * uAdd;

	if(render==true){
		disp = displacement (q, qo, qd, qdo, F, Fo, Ro, alpha, alphaI, beta, gama, eigenVecs, h, u, eigencount, node_count, node_dimensions, block_size, cuda_dat, nodes, fixed_nodes, fixed_nodes_count, scale);

//	free(u);
//	u = new float[node_count*node_dimensions];


	//printf("Address:%l\n",uAdd);


/*Serial Code*/

//	displacement_serial(q, qo,qd, qdo, F, Fo, R, Ro, alpha, alphaI, beta, gama, eigenVecs, u, h, eigencount, node_count, node_dimensions);




	// Copy Old values
	std::copy(qd,qd+eigencount,qdo);
	std::copy(q,q+eigencount,qo);
//	std::copy(R,R+tot_rowCount,Ro);
	std::copy(F,F+((node_count-fixed_nodes_count)*node_dimensions),Fo);


	}

	// TODO add u (displaement)
//	bTestResult = runTest(0, (const char **)"", cuda_dat, u, node_count,scale);
scale=1.0;



//cudaGLUnmapBufferObject(&ptr,BufferId);
cudaGraphicsUnmapResources(1, resources);
 

 
//Destroy the CUDA stream
//cudaStreamDestroy(cuda_stream);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glLoadIdentity();

	glRotatef(angle, 0.0f, 1.0f, 0.0f);
	glRotatef(angleX, 1.0f, 0.0f, 0.0f);

/*	if (ActiveIndexBuffer == 0) {
		glDrawElements(GL_TRIANGLES, 48, GL_UNSIGNED_BYTE, NULL);
	} else {
		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_BYTE, NULL);
	}
*/

	//glDrawElements(GL_TRIANGLES, 48, GL_UNSIGNED_BYTE, NULL);





glColor3f(red,green,blue);
// glDrawElements(GL_QUAD_STRIP, elem_count*elem_nodes*sizeof(elem[0]), GL_UNSIGNED_SHORT, NULL);
glDrawElements(GL_TRIANGLE_STRIP, elem_count*elem_nodes, GL_UNSIGNED_SHORT, NULL);
//glDrawElements(GL_TRIANGLE_STRIP, sizeBuffer/sizeof(GLshort), GL_UNSIGNED_SHORT, NULL);

 glColor3f(1.0f,1.0f,1.0f);
//glDrawElements(GL_LINE_LOOP, elem_count*elem_nodes*sizeof(elem[0]), GL_UNSIGNED_SHORT, NULL);
glDrawElements(GL_LINE_LOOP, elem_count*elem_nodes, GL_UNSIGNED_SHORT, NULL);
//glDrawElements(GL_LINES, sizeBuffer/sizeof(GLshort), GL_UNSIGNED_SHORT, NULL);


	glutSwapBuffers();
	glutPostRedisplay();
}
void Cleanup(void)
{
	DestroyShaders();
	DestroyVBO();
}

/*

void renderScene(void) {

GLuint vertexArray;
glGenBuffers( 1,&vertexArray);
glBindBuffer( GL_ARRAY_BUFFER, vertexArray);
glBufferData( GL_ARRAY_BUFFER, sizeof(nodes), NULL,GL_DYNAMIC_COPY );
cudaGLRegisterBufferObject( vertexArray );


void * vertexPointer;
// Map the buffer to CUDA
cudaGLMapBufferObject(&vertexPointer, vertexArray);
// Run a kernel to create/manipulate the data
//MakeVerticiesKernel<<<gridSz,blockSz>>>(ptr,numVerticies);
bTestResult = runTest(0, (const char **)"", nodes, node_count);
// Unmap the buffer
//cudaGLUnmapBufferObject(vertexArray);
cudaGLUnmapBufferObject(vertexArray);



// Bind the Buffer
glBindBuffer( GL_ARRAY_BUFFER, vertexArray );
// Enable Vertex and Color arrays
glEnableClientState( GL_VERTEX_ARRAY );
//glEnableClientState( GL_COLOR_ARRAY );
// Set the pointers to the vertices and colors
glVertexPointer(3,GL_FLOAT,sizeof(nodes)/node_count,0);
//glColorPointer(4,GL_UNSIGNED_BYTE,16,12);


glColor3f(red,green,blue);
glDrawElements(GL_QUADS, elem_count*elem_nodes, GL_UNSIGNED_SHORT,elem);


//    bTestResult = runTest(0, (const char **)"", nodes, node_count);





/*

		GLuint bufferID;
// Generate a buffer ID
	glGenBuffers(1,&bufferID);
// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
// Allocate data for the buffer
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 640 * 480 * 4,
				 NULL, GL_DYNAMIC_COPY);

	cudaGLRegisterBufferObject( bufferID );




	GLuint textureID;
// Enable Texturing
	glEnable(GL_TEXTURE_2D);
// Generate a texture ID
	glGenTextures(1,&textureID);
// Make this the current texture (remember that GL is state-based)
	glBindTexture( GL_TEXTURE_2D, textureID);
// Allocate the texture memory. The last parameter is NULL since we only
// want to allocate memory, not initialize it
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, 640, 480, 0, GL_BGRA,
				  GL_UNSIGNED_BYTE, NULL);
// Must set the filter mode, GL_LINEAR enables interpolation when scaling
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);


*/





	

//	glColor3f(red,green,blue);

//	GLfloat a[8][3] = {{1,1,1}, { -1,1,1},  {-1,-1,1},  {1,-1,1},        // v0-v1-v2-v3
//					   {1,-1,-1},  {1,1,-1}, {-1,1,-1}, {-1,-1,-1}};

// GLfloat a[] = {1,1,1,  -1,1,1,  -1,-1,1,  1,-1,1,        // v0-v1-v2-v3
//                       1,-1,-1,  1,1,-1, -1,1,-1, -1,-1,-1};

//GLfloat colors[] = {1,1,1,  1,1,0,  1,0,0,  1,0,1,              // v0-v1-v2-v3
//                    1,1,1,  1,0,1,  0,0,1,  0,1,1};


//GLfloat a[]={1,1,1,  -1,1,1, -1,-1,1 ,1,-1,1,
//			 1,1,1,  1,0,1,  0,0,1,  0,1,1,
//			 1,1,1,  1,1,-1,  -1,1,-1,  -1,1,1,        // v0-v5-v6-v1
//			 -1,1,1,  -1,1,-1,  -1,-1,-1,  -1,-1,1,    // v1-v6-v7-v2
//			 -1,-1,-1,  1,-1,-1,  1,-1,1,  -1,-1,1,    // v7-v4-v3-v2
//			 1,-1,-1,  -1,-1,-1,  -1,1,-1,  1,1,-1};


//GLfloat colors[] = {1,1,1,  1,1,0,  1,0,0,  1,0,1,              // v0-v1-v2-v3
//                    1,1,1,  1,0,1,  0,0,1,  0,1,1,              // v0-v3-v4-v5
//                    1,1,1,  0,1,1,  0,1,0,  1,1,0,              // v0-v5-v6-v1
//                    1,1,0,  0,1,0,  0,0,0,  1,0,0,              // v1-v6-v7-v2
//                    0,0,0,  0,0,1,  1,0,1,  1,0,0,              // v7-v4-v3-v2
//                    0,0,1,  0,0,0,  0,1,0,  0,1,1};  

//elem=new GLshort*[6];
//						for(int i=0; i<6; i++)
//						{
//							elem[i] = new GLshort[4];
//						}

//						GLshort el[6][4]={{0,1,2,3},
//										{0,5,4,3},
//										{5,6,7,4},
//										{6,1,2,7},
//										{2,7,4,3},
//										{1,6,5,0}};

//GLshort indx[]={0,1,2,3,
//                     0,5,4,3,
//                     5,6,7,4,
//                     6,1,2,7,
//                     2,7,4,3,
//                     1,6,5,0};




	// ######----------Tetrahedron sort of works---------########


	// GLfloat colors[node_count*node_dimensions];
	// for(int i=0 ;i<node_count*node_dimensions;i++){
	// 	srand (time(0));
	// 	colors[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);;
	// }

//--------OLD DRAW BEGINS-------------
/*
	angle+=deltaAngle;
	angleX+=deltaAngleX;

	// Background Color
//	glClearColor(1.0f, 1.0f, 1.0f, 1.5f);


	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Reset transformations
	glLoadIdentity();

	// Set the camera
	gluLookAt(	0.0f, 0.0f, 10.0f,
			0.0f, 0.0f,  0.0f,
			0.0f, 1.0f,  0.0f);

	glRotatef(angle, 0.0f, 1.0f, 0.0f);
	glRotatef(angleX, 1.0f, 0.0f, 0.0f);



 // glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);


//	glNormalPointer(GL_FLOAT, 0, normals);
   // glColorPointer(3, GL_FLOAT, 0, colors);
    glVertexPointer(3, GL_FLOAT, 0, nodes);


	glPushMatrix();


 //glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);


//	glNormalPointer(GL_FLOAT, 0, normals);
  //  glColorPointer(3, GL_FLOAT, 0, colors);
    glVertexPointer(3, GL_FLOAT, 0, nodes);


	glPushMatrix();
glColor3f(red,green,blue);
	glDrawElements(GL_QUADS, elem_count*elem_nodes, GL_UNSIGNED_SHORT, elem);
glColor3f(1.0f,1.0f,1.0f);
	glDrawElements(GL_LINES, elem_count*elem_nodes, GL_UNSIGNED_SHORT, elem);


    glPopMatrix();

    glDisableClientState(GL_VERTEX_ARRAY);  // disable vertex arrays
    glDisableClientState(GL_COLOR_ARRAY);

*/


//------------OLD DRAW ENDS-----------

//	glBegin(GL_TRIANGLES);
//		glVertex3f(-2.0f,-2.0f, 0.0f);
//		glVertex3f( 2.0f, 0.0f, 0.0);
//		glVertex3f( 0.0f, 2.0f, 0.0);
//	glEnd();
	// ########------Tetrahedron ends-------#######
//	glutSwapBuffers();
//}

void CreateVBO_CUDA(void)
{


GLenum ErrorCheckValue = glGetError();



	BufferSize = sizeof(float)*node_count*node_dimensions; // 544: 32 per element
//	BufferSize=sizeof(Vertices); // Square: 96: 24 Nodes * 4 bytes (float) each
	//const size_t VertexSize = sizeof(Vertices[0]); // 32 per elemen
	VertexSize = sizeof(nodes[0])*node_dimensions; // Square: 12: 4 bytes (float) * node_dimension (3)
//	VertexSize = sizeof(nodes[0])*node_dimensions; // 32 per element
	//const size_t RgbOffset = sizeof(Vertices[0].XYZW); // 16 offset
	
	glGenVertexArrays(1, &VaoId);
	glBindVertexArray(VaoId);
	
	glGenBuffers(1, &BufferId);
	glBindBuffer(GL_ARRAY_BUFFER, BufferId);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, nodes, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset);
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);

	glEnableVertexAttribArray(0);
//	glEnableVertexAttribArray(1);



//Register Pixel Buffer Object as CUDA graphics resource
cudaGraphicsGLRegisterBuffer(resources, BufferId, cudaGraphicsMapFlagsNone);







//let's have a look at the buffer
    // glBindBuffer(GL_ARRAY_BUFFER, BufferId);
    // float* mappedBuffer = (float *) glMapBuffer(GL_ARRAY_BUFFER,GL_READ_ONLY);
    // printf("\tbefore mapping:\n");
	// 	   for(int h=0;h<node_count*node_dimensions;h++){
	// 		   printf("Index: %d, Value: %f\n",h,mappedBuffer[h]);
	// 	   }
    // glUnmapBuffer(GL_ARRAY_BUFFER);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);


//cudaStream_t cuda_stream;
 
//Create CUDA stream
//Might not be necessary
//cudaStreamCreate(&cuda_stream);
 
//Map the graphics resource to the CUDA stream
//cudaGraphicsMapResources(1, resources, cuda_stream);
// map and unmap the cuda resource
// if (cudaGraphicsMapResources(1, resources,0) != cudaSuccess)
//         printf("Resource mapping failed...\n");
//    if (cudaGraphicsUnmapResources(1, &resources, 0) != cudaSuccess)
//        printf("Resource unmapping failed...\n");



    // // let's have a look at the buffer again
    // glBindBuffer(GL_ARRAY_BUFFER, BufferId);
    // mappedBuffer = (int *) glMapBuffer(GL_ARRAY_BUFFER,GL_READ_ONLY);
    // printf("\tafter mapping: %d, %d, %d\n",mappedBuffer[0], mappedBuffer[1], 
    //         mappedBuffer[2]);
    // glUnmapBuffer(GL_ARRAY_BUFFER);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);

 
//Call CUDA function
//map_texture(cuda_dat, siz, resources[0]);
//Unmap the CUDA stream
//cudaGraphicsUnmapResources(1, resources,cuda_stream);
 
//Destroy the CUDA stream
//cudaStreamDestroy(cuda_stream);

//	cudaGLRegisterBufferObject( BufferId );


	glGenBuffers(2, IndexBufferId);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[0]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, elem_count*elem_nodes*sizeof(elem),elem, GL_STATIC_DRAW);

	glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &sizeBuffer);



/*	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(AlternateIndices), AlternateIndices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[0]);
*/
	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR)
	{
		fprintf(
			stderr,
			"ERROR: Could not create a VBO: %s \n",
			gluErrorString(ErrorCheckValue)
		);

		exit(-1);
	}

}

void CreateVBO(void)
{
GLenum ErrorCheckValue = glGetError();
	BufferSize = sizeof(float)*node_count*node_dimensions; // 544: 32 per element
//	BufferSize=sizeof(Vertices); // Square: 96: 24 Nodes * 4 bytes (float) each
	//const size_t VertexSize = sizeof(Vertices[0]); // 32 per elemen
	VertexSize = sizeof(nodes[0])*node_dimensions; // Square: 12: 4 bytes (float) * node_dimension (3)
//	VertexSize = sizeof(nodes[0])*node_dimensions; // 32 per element
	//const size_t RgbOffset = sizeof(Vertices[0].XYZW); // 16 offset
	
	glGenVertexArrays(1, &VaoId);
	glBindVertexArray(VaoId);
	
	glGenBuffers(1, &BufferId);
	glBindBuffer(GL_ARRAY_BUFFER, BufferId);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, nodes, GL_DYNAMIC_COPY);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);

	glEnableVertexAttribArray(0);
//	glEnableVertexAttribArray(1);

	glGenBuffers(2, IndexBufferId);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[0]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, elem_count*elem_nodes*sizeof(elem),elem, GL_STATIC_DRAW);



/*	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(AlternateIndices), AlternateIndices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[0]);
*/
	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR)
	{
		fprintf(
			stderr,
			"ERROR: Could not create a VBO: %s \n",
			gluErrorString(ErrorCheckValue)
		);

		exit(-1);
	}

}
void DestroyVBO(void)
{
	GLenum ErrorCheckValue = glGetError();

	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &BufferId);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDeleteBuffers(2, IndexBufferId);

	glBindVertexArray(0);
	glDeleteVertexArrays(1, &VaoId);

	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR)
	{
		fprintf(
			stderr,
			"ERROR: Could not destroy the VBO: %s \n",
			gluErrorString(ErrorCheckValue)
		);

		exit(-1);
	}

}

void CreateShaders(void)
{
	GLenum ErrorCheckValue = glGetError();
	
	VertexShaderId = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(VertexShaderId, 1, &VertexShader, NULL);
	glCompileShader(VertexShaderId);

	FragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(FragmentShaderId, 1, &FragmentShader, NULL);
	glCompileShader(FragmentShaderId);

	ProgramId = glCreateProgram();
		glAttachShader(ProgramId, VertexShaderId);
		glAttachShader(ProgramId, FragmentShaderId);
	glLinkProgram(ProgramId);
	glUseProgram(ProgramId);

	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR)
	{
		fprintf(
			stderr,
			"ERROR: Could not create the shaders: %s \n",
			gluErrorString(ErrorCheckValue)
		);

		exit(-1);
	}
}

void DestroyShaders(void)
{
	GLenum ErrorCheckValue = glGetError();

	glUseProgram(0);

	glDetachShader(ProgramId, VertexShaderId);
	glDetachShader(ProgramId, FragmentShaderId);

	glDeleteShader(FragmentShaderId);
	glDeleteShader(VertexShaderId);

	glDeleteProgram(ProgramId);

	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR)
	{
		fprintf(
			stderr,
			"ERROR: Could not destroy the shaders: %s \n",
			gluErrorString(ErrorCheckValue)
		);

		exit(-1);
	}
}

void processNormalKeys(unsigned char key, int x, int y) {

	if (key == 27)
		exit(0);
	if (key==32){
		if(render==true)
			render=false;
		else
			render=true;
	}
}



void processSpecialKeys(int key, int x, int y) {

	switch(key) {
	case GLUT_KEY_F1 :
		red = 1.0;
		green = 0.0;
		blue = 0.0; 
		break;
	case GLUT_KEY_F2 :
		red = 0.0;
		green = 1.0;
		blue = 0.0; break;
	case GLUT_KEY_F3 :
		red = 0.0;
		green = 0.0;
		blue = 1.0; break;
	case GLUT_KEY_F4 :
		red = 1.0;
		green = 1.0;
		blue = 1.0; break;


		case GLUT_KEY_LEFT : deltaAngle = -0.35f; break;
		case GLUT_KEY_RIGHT : deltaAngle = 0.35f; break;
		case GLUT_KEY_UP : deltaAngleX = -0.35f; break;
		case GLUT_KEY_DOWN : deltaAngleX = 0.35f; break;
	}
}

void releaseKey(int key, int x, int y) {

	switch (key) {
		case GLUT_KEY_LEFT :
		case GLUT_KEY_RIGHT : deltaAngle = 0.0f;break;
		case GLUT_KEY_UP :
		case GLUT_KEY_DOWN : deltaAngleX = 0.0f;break;
	}
}

void mouseMove(int x, int y) { 	

         // this will only be true when the left button is down
         if (xOrigin >= 0) {

		// update deltaAngle
		deltaAngle = (x - xOrigin) * 0.01f;
		deltaAngleX = (y - yOrigin) * 0.01f;

		// update camera's direction
		//lx = sin(angle + deltaAngle);
		//lz = -cos(angle + deltaAngle);
	}
}

void mouseButton(int button, int state, int x, int y) {


// Wheel reports as button 3(scroll up) and button 4(scroll down)
	if ((button == 3) || (button == 4)) // It's a wheel event
	{
		// Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
//	       if (state == GLUT_UP) return; // Disregard redundant GLUT_UP events
//	       printf("Scroll %s At %d %d\n", (button == 3) ? "Up" : "Down", x, y);
		//if (state == GLUT_UP)
		//	deltaMove = 0.0f;	
		//else{
		if(button==3){
			//deltaMove = 3.0f;
			//x += deltaMove * lx * 0.1f;
			//z += deltaMove * lz * 0.1f;
			scale=1.2;
		}
		else if(button==4){
			//deltaMove = -3.0f;
			//x += deltaMove * lx * 0.1f;
			//z += deltaMove * lz * 0.1f;
			scale=0.8;
		}
	}

	//else{  // normal button event
	//       printf("Button %s At %d %d\n", (state == GLUT_DOWN) ? "Down" : "Up", x, y);
	   //  }
	// only start motion if the left button is pressed

	if (button == GLUT_LEFT_BUTTON) {

		// when the button is released
		if (state == GLUT_UP) {
			deltaAngle=0;
			deltaAngleX=0;
			//angle += deltaAngle;
			xOrigin = -1;
			yOrigin=-1;
		}
		else  {// state = GLUT_DOWN
			xOrigin = x;
			yOrigin= y;
			//deltaAngle=0;
		}
	}
}





int main(int argc, char **argv)
{

/*#########---------OpenGL Init------------#########*/


	// init GLUT and create window
	glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(640,480);
	glutCreateWindow("Test_TetgenModelRenderer_OpenGL_CUDA");

	// register callbacks
	glutDisplayFunc(RenderFunction);
	glutReshapeFunc(ResizeFunction);
	glutIdleFunc(IdleFunction);
	glutCloseFunc(Cleanup);

//	glutDisplayFunc(renderScene);
//	glutReshapeFunc(changeSize);
//	glutIdleFunc(renderScene);

	// here are the new entries
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);
	glutIgnoreKeyRepeat(1);
	glutSpecialUpFunc(releaseKey);



	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);


	GLenum GlewInitResult;

	glewExperimental = GL_TRUE;
	GlewInitResult = glewInit();

	if (GLEW_OK != GlewInitResult) {
		fprintf(
			stderr,
			"ERROR: %s\n",
			glewGetErrorString(GlewInitResult)
		);
		exit(EXIT_FAILURE);
	}
	
	fprintf(
		stdout,
		"INFO: OpenGL Version: %s\n",
		glGetString(GL_VERSION)
	);

	//cudaSetDevice(0);
	//cudaGLSetGLDevice(0);

	//CreateShaders();
	//CreateVBO_CUDA();
	glClearColor(0.25f, 0.25f, 0.25f, 0.0f);


/*#########---------Ends OpenGL Init------------#########*/



/*#########---------Starts Getopts code------------#########*/


// Extern because vars declared into getopts
	extern char *optarg;
	extern int optind;
	int c, err = 0; 
	int nflag=0, kflag=0;
	string ename,nname,kname,eigenvec_name, fixedname;
	static char usage[] = "usage: %s -n Node_filename [-k eigen_filename] \n";

	while ((c = getopt(argc, argv, "n:k:")) != -1)
		switch (c) {
		case 'n':
			nflag = 1;
			// Convert to strings & append extensions
			nname.append(optarg);
			nname.append(".node");
			ename.append(optarg);
			ename.append(".ele");
			kname.append(optarg);
			kname.append(".csv");
			eigenvec_name.append(optarg);
			eigenvec_name.append("_vec.csv");
			fixedname.append(optarg);
			fixedname.append("_fixed.csv");
			break;
		case 'k':
			kflag = 1;
			// Convert to string & append extension
			kname.append(optarg);
			kname.append(".csv");
			break;
		case '?':
			err = 1;
			break;
		}
	if (nflag == 0) 
	{	/* -n (Node_filename is mandatory */
		fprintf(stderr, "%s: Node_filename is missing \n", argv[0]);
		fprintf(stderr, usage, argv[0]);
		exit(1);
	} 
	else if (kflag==0)
	  {
		  // TODO fix nmname to get rid of extension
		  //	kname=nname;
//		kname.append(".csv");
	  }
	else if ((optind+1-1) > argc) 
	{	
		/* More arguments ? */
		printf("optind = %d, argc=%d\n", optind, argc);
		fprintf(stderr, "%s: missing last arguments\n", argv[0]);
		fprintf(stderr, usage, argv[0]);
		exit(1);
	} 
	else if (err) 
	{
		fprintf(stderr, usage, argv[0]);
		exit(1);
	}

	/* Print Values */
	cout<<"NodeFile:"<<nname<<"\n";
	cout<<"ElemFile:"<<ename<<"\n";

	// Test Printf on C++ string
    printf("EigenFile:%s\n", kname.c_str());

	cout<<"EigenVecsFile:"<<eigenvec_name<<"\n";


	
	if (optind < argc)
	{	/* Last arguments */
		for (; optind < argc; optind++)
			printf("argument: \"%s\"\n", argv[optind]);
//	else {
//		printf("no arguments left to process\n");
	}

/* End of Getopts  */

/*#########---------End of Getopts------------#########*/



/*#########---------File Loading------------#########*/


/* Open Files  */

	// Node File
ifstream node_file(nname.c_str());
string line;
	
	int line_count=0;
	int node_start=0;

	//int node_count;
//	int node_dimensions;
	//float **nodes;
	while(std::getline(node_file,line))
	{
// Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream  lineStream(line);
        std::string        cell;
		int val_count=0;
        while(std::getline(lineStream,cell,' '))
        {
			if(cell.length()>0)
			{
				//cout<<val_count;
				// std::stringstream trimmer;
				// trimmer << cell;
				// cell.clear();
				// trimmer >> cell;
				if(line_count==0)
				{
					if(val_count==0)
						// Get Number of Nodes
						node_count=atoi(cell.c_str());
					else if(val_count==1){
						//cout<<cell;
						//Get Number of Dimensions
						node_dimensions=atoi(cell.c_str());
						
						// Allocate multi-dimensional array
//						**nodes = (float **) malloc(node_count*sizeof(float*));

						nodes=new float[node_count*node_dimensions];
						// nodes=new float*[node_count];
						// for(int i=0; i<node_count; i++)
						// {
						// 	nodes[i] = new GLfloat[node_dimensions];
						// }

					}
				}
				else
				{
			
					char* p;
					float converted = strtof(cell.c_str(), &p);
					if (*p) 
					{
						// conversion failed because the input wasn't a number
					}
					else 
					{

						if(line_count==1 & val_count==0)
							node_start=converted;
						if(val_count>0)
						{
							// if(val_count==3)
							// 	converted*=8;
							nodes[node_dimensions*(line_count-1)+(val_count-1)]=converted;
							// nodes[line_count-1][val_count-1]=converted;
							//cout<<nodes[line_count-1][val_count-1]<<",";
						// use converted
						//	cout << converted<<line_count<<"\n";// You have a cell!!!!
						}
					}
				}
				val_count++;
			}
		}
		//cout<<"\n";
		line_count++;
	}


// Elements File
	ifstream elem_file(ename.c_str());
	//string line;
	
	line_count=0;
	//int elem_count;
	//int elem_nodes;
	//int **elem;
	while(std::getline(elem_file,line))
	{
// Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream  lineStream(line);
        std::string        cell;
		int val_count=0;
        while(std::getline(lineStream,cell,' '))
        {
			if(cell.length()>0)
			{
				//cout<<val_count;
				// std::stringstream trimmer;
				// trimmer << cell;
				// cell.clear();
				// trimmer >> cell;
				if(line_count==0)
				{
					if(val_count==0)
						// Get Number of Elements
						elem_count=atoi(cell.c_str());
					else if(val_count==1){
						//Get Number of Element Nodes
						elem_nodes=atoi(cell.c_str());
						
						// Allocate multi-dimensional array
						elem=new GLshort[elem_count*elem_nodes];
						// elem=new GLint*[elem_count];
						// for(int i=0; i<elem_count; i++)
						// {
						// 	elem[i] = new GLint[elem_nodes];
						// }

					}
				}
				else
				{
			
					char* p;
					GLshort converted = strtol(cell.c_str(), &p,10);
					//unsigned short converted = (unsigned short) strtoul(cell.c_str(), NULL, 0);
					if (*p) 
					{
						// conversion failed because the input wasn't a number
					}
					else 
					{
						if(val_count>0)
						{
							elem[elem_nodes*(line_count-1)+(val_count-1)]=converted-node_start;
							// elem[line_count-1][val_count-1]=converted; 
							//cout<<elem[line_count-1][val_count-1]<<",";
						// use converted
							//cout << converted<<"\n";// You have a cell!!!!
						}
					}
				}
				val_count++;
			}
		}
		//cout<<"\n";
		line_count++;
	}

	// EigenFile

	// No need for columns in EIGENVALUES
// Getfile handle
	std::ifstream csv_file_rows(kname.c_str());
	std::ifstream csv_file(kname.c_str());
	unsigned int row_count=0;
	unsigned int tot_rowCount;
	//std::string line;	

// Read whole file to get row and count
	 while (std::getline(csv_file_rows,line)){
	// 	if(row_count==1){
	// 		std::stringstream lineStream(line);
	// 		std::string colVal;
	// 		while(std::getline(lineStream,colVal,',')){
	// 			col_count++;
	// 		}
	// 	}

		row_count++;
		//	std::cout<<line<<"\t";
//		std::cout<<row_count<<"\n";
	}


	// Print debug values
	tot_rowCount=row_count;
//	tot_colCount=col_count;

	// std::cout<<tot_colCount<<"\n";

	// Allocate arrays
	eigenVals=new float[tot_rowCount];
	// eigenVals=new float[tot_rowCount*tot_colCount];

	row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(csv_file,line)){
    // Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;
		
		// Reset col value
		//col_count=0;

		// Separate values by ','
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also
				// Convert char* to float
				char *p;
				float converted = strtof(colVal.c_str(),&p);
				if (*p){}
				else{
					eigenVals[row_count]=converted;
					//std::cout<<eigenVals[row_count]<<"\n";
					// eigenVals[(tot_colCount*row_count)+col_count]=converted;
					// Print it to check if its correct
					// std::cout<<(tot_colCount*row_count)+col_count<<"\t"<<converted<<"\n";
					// std::cout<<eigenVals[(tot_colCount*row_count)+col_count]<<"\t"<<converted<<"\n";
				}
			}
			//	col_count++;
		}
		row_count++;
	}

// EigenVectorsFile
// Getfile handle

	std::ifstream eigenvec_file_rows(eigenvec_name.c_str());
	std::ifstream eigenvec_file(eigenvec_name.c_str());

	// Columns should be same as eigenVals count
	// Rows should be the same as nodes x dimensions
	unsigned int col_countEvec=0;
	unsigned int tot_colCountEvec;

	//std::string line;	

// Read whole file to get row and count
	// while (std::getline(eigenvec_file_rows,line)){
	// 	if(col_countEvec==0){
	// 		std::stringstream lineStream(line);
	// 		std::string colVal;
	// 		while(std::getline(lineStream,colVal,',')){
	// 			col_countEvec++;
	// 		}
	// 	}
	// }
	
// Print debug values
	// Columns same as EigenVals count
	tot_colCountEvec=tot_rowCount;



	// Allocate arrays
	eigenVecs=new float[node_count*node_dimensions*tot_colCountEvec];

	row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(eigenvec_file,line)){
    // Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;
		
		// Reset col value
		col_countEvec=0;

		// Separate values by ','
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also
				// Convert char* to float
				char *p;
				float converted = strtof(colVal.c_str(),&p);
				if (*p){}
				else{
					eigenVecs[(tot_colCountEvec*row_count)+col_countEvec]=converted;
					// Print it to check if its correct
					//std::cout<<eigenVecs[(node_count*node_dimensions*row_count)+col_countEvec]<<",";
					//std::cout<<eigenVals[(tot_colCountEvec*row_count)+col_countEvec]<<"\n";
					//std::cout<<(tot_colCountEvec*row_count)+col_countEvec<<"\n";
				}
			}
			col_countEvec++;
		}
		row_count++;
	}



// Fixed Nodes


	// No need for columns in Fixed Nodes
// Getfile handle
	std::ifstream fixed_nodes_rows(fixedname.c_str());
	std::ifstream fixed_nodes_file(fixedname.c_str());
    row_count=0;
	//std::string line;	

// Read whole file to get row and count
	 while (std::getline(fixed_nodes_rows,line)){
	// 	if(row_count==1){
	// 		std::stringstream lineStream(line);
	// 		std::string colVal;
	// 		while(std::getline(lineStream,colVal,',')){
	// 			col_count++;
	// 		}
	// 	}

		row_count++;
		//	std::cout<<line<<"\t";
//		std::cout<<row_count<<"\n";
	}


	// Print debug values
	fixed_nodes_count=row_count;
//	tot_colCount=col_count;

	// std::cout<<tot_colCount<<"\n";

	// Allocate arrays
	fixed_nodes=new int[fixed_nodes_count];
	// eigenVals=new float[tot_rowCount*tot_colCount];

	row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(fixed_nodes_file,line)){
    // Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;
		
		// Reset col value
		//col_count=0;

		// Separate values by ','
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also
				// Convert char* to float
				char *p;
				int converted = strtol(colVal.c_str(),&p,10);
				if (*p){}
				else{
					fixed_nodes[row_count]=converted-1;
					//std::cout<<eigenVals[row_count]<<"\n";
					// eigenVals[(tot_colCount*row_count)+col_count]=converted;
					// Print it to check if its correct
					// std::cout<<(tot_colCount*row_count)+col_count<<"\t"<<converted<<"\n";
					// std::cout<<eigenVals[(tot_colCount*row_count)+col_count]<<"\t"<<converted<<"\n";
				}
			}
			//	col_count++;
		}
		row_count++;
	}



/*#########---------Ends File Loading---------------#########*/



/*#########---------Allocate coefficient matrices---------------#########*/

	// D, alpha, beta, gamma square matrices eigencount x eigencount

	// Declare h
	h=0.03;

	d = new float[tot_rowCount*tot_rowCount];
	alpha = new float[tot_rowCount*tot_rowCount];
	beta = new float[tot_rowCount*tot_rowCount];
	gama = new float[tot_rowCount*tot_rowCount];
	M =  new float[tot_rowCount*tot_rowCount];
	// C can be a vector constants: M*0.01+K*0.001
	C =  new float[tot_rowCount];

	// Create alpha-identity
	alphaI = new float[tot_rowCount*tot_rowCount];


	// 2 nested fors to declare square diagonal matrices
	for(int i=0;i<tot_rowCount;i++){
		// Maybe only need 1 for ?
		for(int j=0;j<tot_rowCount;j++){
			int index=i*tot_rowCount+j;
			// Only declare diagonal elements
			if(i==j){
				M[index]=1;
				C[i]=0.01+0.001*eigenVals[i];
				d[index]=1+h*C[i]+eigenVals[i]*h*h;
				alpha[index]=1-((h*h)*eigenVals[i]/d[index]);
				beta[index]=h*(1-(h*C[i]+(h*h)*eigenVals[i])/d[index]);
				gama[index]=h*h/d[index];
				alphaI[index]=alpha[index]-1;
			}
			else{
				M[index]=0;
				d[index]=0;
				alpha[index]=0;
				beta[index]=0;
				gama[index]=0;
				alphaI[index]=0;
			}
			//std::cout<<gama[i*tot_rowCount+j]<<",";
		}
	}


/*#########---------Ends Coefficients allocation---------------#########*/





/*#########---------Allocate calculation vectors---------------#########*/


// Allocate vectors
// F and u node dimensioned all others EigenVals dimensioned
	F = new float[(node_count-fixed_nodes_count)*node_dimensions];
	Fo = new float[(node_count-fixed_nodes_count)*node_dimensions];
	q = new float[tot_rowCount];
	qo = new float[tot_rowCount];
	qd = new float[tot_rowCount];
	qdo = new float[tot_rowCount];
	u = new float[node_count*node_dimensions];
	R = new float[node_count*node_dimensions*node_count*node_dimensions];
	Ro = new float[node_count*node_dimensions*node_count*node_dimensions];



	// Some force constant
	const float force=0.03;
	// Delcare F in only 1 direction
	for (int i=0;i<(node_count-fixed_nodes_count)*node_dimensions;i++){
		if(3-(i%3)==1)
			F[i]=force;
		else
			F[i]=0;
		// F[i]=force;
		u[i]=0;
		//	std::cout<<F[i]<<",";
	}

	// Copy F to Fo
	std::copy(F,F+((node_count-fixed_nodes_count)*node_dimensions),Fo);

	// For loop to initialize all other vectors (all 0's)
	for (int i=0;i<tot_rowCount;i++){
		q[i]=0;
		qo[i]=0;
		qd[i]=0;
		qdo[i]=0;
	}




/*#########---------Ends calculation vectors allocation---------------#########*/




	CreateVBO_CUDA();
//	CreateShaders();
	
	// No sure I should call this here, is it necessary?
    // run the device part of the program
//    bTestResult = runTest(argc, (const char **)argv, nodes, node_count,1.0);

/*
	for (int ii=0;ii<node_count*3;ii++)
	{
		cout<<nodes[ii]<<"\n";
	}
*/

	eigencount=tot_rowCount;

	cout<<"Nodes: "<<node_count<<"\t";
	cout<<"Dimensions: "<<node_dimensions<<"\n";
	cout<<"Elements: "<<elem_count<<"\t";
	cout<<"Nodes per Element: "<<elem_nodes<<"\n";
	std::cout<<"# Eigenvalues: "<<tot_rowCount<<"\n";
	std::cout<<"Eigenvectors matrix dimensions:"<<node_count*node_dimensions<<"x"<<tot_rowCount<<"\n";

//	int abc=1;
//	bool a = runTest2(0, (const char **)"", abc);

//	allocate_GPUnodes(d_nodes, nodes, node_count, node_dimensions);
	


	glutMainLoop();

//	free_GPUnodes(d_nodes);

	cudaDeviceReset();



/*#########---------Free Memory---------------#########*/


	free(nodes);
	free(elem);
	free(eigenVals);
	free(eigenVecs);
	free(d);
	free(alpha);
	free(alphaI);
	free(beta);
	free(gama);
	free(M);
	free(C);
	free(F);
	free(Fo);
	free(q);
	free(qo);
	free(qd);
	free(qdo);
	free(u);



/*#########---------Free Memory---------------#########*/


	return 1;
    //exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);

}
