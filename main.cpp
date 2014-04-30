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
#include <time.h>       /* time */

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
//#define GL_GLEXT_PROTOTYPES


// Buffer stuff
//#include <GL/glew.h>  

//#include "glInfo.h" 
//#include "glext.h" 


/* CUDA Includes */
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_gl_interop.h>

using namespace std;



float *nodes;
GLshort *elem;

int elem_count, elem_nodes;
int node_count, node_dimensions;


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

extern "C" bool runTest2(const int argc, const char **argv, int i);
	bool bTestResult;
extern "C" bool runTest(const int argc, const char **argv, float * buffer, int node_count, float scale);
extern "C" void map_texture(void *cuda_data, size_t size,cudaGraphicsResource *resource);


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
	gluPerspective(45.0f, ratio, 0.1f, 100.0f);
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
 
//Map the graphics resource to the CUDA stream
if (cudaGraphicsMapResources(1, resources,0) != cudaSuccess)
        printf("Resource mapping failed...\n");
if(cudaGraphicsResourceGetMappedPointer((void**)&cuda_dat, &siz, *resources) !=cudaSuccess)
	    printf("Resource pointer mapping failed...\n");



bTestResult = runTest(0, (const char **)"", cuda_dat, node_count,scale);
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

glColor3f(1.0f,1.0f,0.0f);
glDrawElements(GL_QUAD_STRIP, elem_count*elem_nodes*sizeof(elem[0]), GL_UNSIGNED_SHORT, NULL);
 glColor3f(1.0f,1.0f,1.0f);
glDrawElements(GL_LINE_LOOP, elem_count*elem_nodes*sizeof(elem[0]), GL_UNSIGNED_SHORT, NULL);


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
	glBufferData(GL_ARRAY_BUFFER, BufferSize, nodes, GL_DYNAMIC_COPY);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset);
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);

	glEnableVertexAttribArray(0);
//	glEnableVertexAttribArray(1);



//Register Pixel Buffer Object as CUDA graphics resource
cudaGraphicsGLRegisterBuffer(resources, BufferId, cudaGraphicsMapFlagsNone);




// let's have a look at the buffer
    // glBindBuffer(GL_ARRAY_BUFFER, BufferId);
    // int* mappedBuffer = (int *) glMapBuffer(GL_ARRAY_BUFFER,GL_READ_ONLY);
    // printf("\tbefore mapping: %d, %d, %d\n",mappedBuffer[0], mappedBuffer[1], 
    //         mappedBuffer[2]);
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
}



void processSpecialKeys(int key, int x, int y) {

	switch(key) {
		case GLUT_KEY_F1 :
				red = 1.0;
				green = 0.0;
				blue = 0.0; break;
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
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);




// Extern because vars declared into getopts
	extern char *optarg;
	extern int optind;
	int c, err = 0; 
	int nflag=0, kflag=0;
	string ename,nname,kname;
	static char usage[] = "usage: %s -n Node_filename [-k eigen_filename] \n";

	// Getopts
	while ((c = getopt(argc, argv, "n:k:")) != -1)
		switch (c) {
		case 'n':
			nflag = 1;
			// Convert to strings & append extensions
			nname.append(optarg);
			nname.append(".node");
			ename.append(optarg);
			ename.append(".ele");
			break;
		case 'k':
			kflag = 1;
			// Convert to string & append extension
			kname.append(optarg);
			kname.append(".mat");
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
		kname=nname;
		kname.append(".mat");
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
	
	if (optind < argc)
	{	/* Last arguments */
		for (; optind < argc; optind++)
			printf("argument: \"%s\"\n", argv[optind]);
//	else {
//		printf("no arguments left to process\n");
	}

/* End of Getopts  */

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

	cout<<"Nodes: "<<node_count<<"\n";
	cout<<"Dimensions: "<<node_dimensions<<"\n";
	cout<<"Elements: "<<elem_count<<"\n";
	cout<<"Nodes per Element: "<<elem_nodes<<"\n";

//	int abc=1;
//	bool a = runTest2(0, (const char **)"", abc);
	


	glutMainLoop();

	cudaDeviceReset();


	return 1;
    //exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);

}

