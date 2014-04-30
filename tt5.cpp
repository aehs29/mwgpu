#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <math.h>
#include <unistd.h>
#include <iomanip>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


void Cleanup(void);
void CreateVBO(void);
void DestroyVBO(void);
void CreateShaders(void);
void DestroyShaders(void);
void changeSize(int w, int h);
void IdleFunction(void);
void renderScene(void);

float red=1.0f, blue=0.0f, green=0.0f;

// angle for rotating triangle
float angle = 0.0f;
float deltaAngle = 0.0f;
int xOrigin = -1;

float angleX = 0.0f;
float deltaAngleX = 0.0f;
int yOrigin=-1;


// actual vector representing the camera's direction
float lx=0.0f,lz=-1.0f;
// XZ position of the camera
float x=0.0f, z=5.0f;
float deltaMove = 0;


float scale=1.0;


size_t BufferSize,VertexSize;


// float **nodes;
// GLshort **elem;

float *nodes;
GLshort *elem;

int elem_count, elem_nodes;
int node_count, node_dimensions;

int CurrentWidth = 600,
	CurrentHeight = 480,
	WindowHandle = 0;
unsigned FrameCount = 0;
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

/*
const GLchar* VertexShader =
{
	"layout(location=0) in vec4 in_Color;\n"\
	"out vec4 ex_Color;\n"\

	"void main(void)\n"\
	"{\n"\
	"	gl_Position = in_Position;\n"\
	"	ex_Color = in_Color;\n"\
	"}\n"
};
*/

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

typedef struct
{
	float XYZW[4];
//	float RGBA[4];
} Vertex;

/*
GLfloat Vertices[] ={ 0.5f, 0.5f, 0.5f , // 0

		-0.5f, 0.5f, 0.5f , // 0
		-0.5f, -0.5f, 0.5f, // 0
		0.5f, -0.5f, 0.5f , // 0
		0.5f, -0.5f, -0.5f, // 0
		0.5f, 0.5f, -0.5f , // 0
		-0.5f, 0.5f, -0.5f, // 0
		-0.5f, -0.5f, -0.5f  // 0
	};


*/



void computePos(float deltaMove) {

	x += deltaMove * lx * 0.1f;
	z += deltaMove * lz * 0.1f;
}

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
//	gluPerspective(45.0f, ratio, 0.1f, 100.0f);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);
}

void IdleFunction(void)
{
	glutPostRedisplay();
}


void Modify(void){
	
int i=0;
if(scale!=1.0){
	for (i=0;i<node_count*node_dimensions;i++){
		nodes[i]=nodes[i]*scale;
		//Vertices[i]=Vertices[i]*scale*0.9991;
	}
	scale=1.0;
}


glBindBuffer(GL_ARRAY_BUFFER, BufferId);
//    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertexPositions), &fNewData[0]);
glBufferSubData(GL_ARRAY_BUFFER,0, BufferSize, nodes);   
//glBufferData(GL_ARRAY_BUFFER, BufferSize, Vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
	//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset);

	glEnableVertexAttribArray(0);   

}

void RenderFunction(void)
{

	if (deltaMove)
		computePos(deltaMove);
	//++FrameCount;
	angle+=deltaAngle;
	angleX+=deltaAngleX;
	
	Modify();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glLoadIdentity();

	// gluLookAt(	0.0f, 0.0f, 1.0f,
	// 		0.0f, 0.0f,  0.0f,
	// 		0.0f, 1.0f,  0.0f);
	//gluLookAt(	x, 1.0f, z,x+lx, 1.0f,  z+lz,0.0f, 1.0f,  0.0f);

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

void CreateVBO(void)
{
/*
	Vertex Vertices[] =
	{
		{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 0

		// Top
		{ { -0.2f, 0.8f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } }, // 1
		{ { 0.2f, 0.8f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }, // 2
		{ { 0.0f, 0.8f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f } }, //3
		{ { 0.0f, 1.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },	// 4

		// Bottom
		{ { -0.2f, -0.8f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }, // 5
		{ { 0.2f, -0.8f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } }, // 6
		{ { 0.0f, -0.8f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f } }, //7
		{ { 0.0f, -1.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },	// 8

		// Left
		{ { -0.8f, -0.2f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } }, // 9
		{ { -0.8f, 0.2f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }, // 10
		{ { -0.8f, 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f } }, //11
		{ { -1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },	// 12

		// Right
		{ { 0.8f, -0.2f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }, // 13
		{ { 0.8f, 0.2f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } }, // 14
		{ { 0.8f, 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f } }, //15
		{ { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } }	// 16
	};
*/

/*
	Vertex Vertices[] =
	{
		{ { 0.0f, 0.0f, 0.0f, 1.0f } }, // 0

		// Top
		{ { -0.2f, 0.8f, 0.0f, 1.0f } }, // 1
		{ { 0.2f, 0.8f, 0.0f, 1.0f } }, // 2
		{ { 0.0f, 0.8f, 0.0f, 1.0f } }, //3
		{ { 0.0f, 1.0f, 0.0f, 1.0f } },	// 4

		// Bottom
		{ { -0.2f, -0.8f, 0.0f, 1.0f } }, // 5
		{ { 0.2f, -0.8f, 0.0f, 1.0f } }, // 6
		{ { 0.0f, -0.8f, 0.0f, 1.0f } }, //7
		{ { 0.0f, -1.0f, 0.0f, 1.0f } },	// 8

		// Left
		{ { -0.8f, -0.2f, 0.0f, 1.0f } }, // 9
		{ { -0.8f, 0.2f, 0.0f, 1.0f } }, // 10
		{ { -0.8f, 0.0f, 0.0f, 1.0f } }, //11
		{ { -1.0f, 0.0f, 0.0f, 1.0f } },	// 12

		// Right
		{ { 0.8f, -0.2f, 0.0f, 1.0f } }, // 13
		{ { 0.8f, 0.2f, 0.0f, 1.0f } }, // 14
		{ { 0.8f, 0.0f, 0.0f, 1.0f } }, //15
		{ { 1.0f, 0.0f, 0.0f, 1.0f } }	// 16

	};
*/
	
/*
	GLshort Indices[] = {
		// Top
		0, 1, 3,
		0, 3, 2,
		3, 1, 4,
		3, 4, 2,

		
		// Bottom
		0, 5, 7,
		0, 7, 6,
		7, 5, 8,
		7, 8, 6,

		// Left
		0, 9, 11,
		0, 11, 10,
		11, 9, 12,
		11, 12, 10,

		// Right
		0, 13, 15,
		0, 15, 14,
		15, 13, 16,
		15, 16, 14
		};

*/
/*
	Vertex Vertices[]={
		{{1.0f,1.0f,1.0f,1.0f}},
		{ { -1.0f, 1.0f, 1.0f, 1.0f } }, // 1
		{ { -1.0f, -1.0f, 1.0f, 1.0f } },
		{ { 1.0f, -1.0f, 1.0f ,1.0f} }, // 0
		{ { 1.0f, -1.0f, -1.0f ,1.0f} }, // 0
		{ { 1.0f, 1.0f, -1.0f ,1.0f} }, // 0
		{ { -1.0f, 1.0f, -1.0f,1.0f } }, // 0
		{ { -1.0f, -1.0f, -1.0f ,1.0f} } // 0

};
*/

/*
GLshort Indices[] = {
		// Top
	0, 1, 2,3,
	0,5,4,3,
	5,6,7,4,
	6,1,2,7,
	2,7,4,3,
	1,6,5,0
};

*/


/*
	Vertex Vertices[] =
	{
		{ { 0.5f, 0.5f, 0.5f }}, // 0

		{ { -0.5f, 0.5f, 0.5f } }, // 0
		{ { -0.5f, -0.5f, 0.5f }}, // 0
		{ { 0.5f, -0.5f, 0.5f } }, // 0
		{ { 0.5f, -0.5f, -0.5f } }, // 0
		{ { 0.5f, 0.5f, -0.5f } }, // 0
		{ { -0.5f, 0.5f, -0.5f} }, // 0
		{ { -0.5f, -0.5f, -0.5f } } // 0
	};

*/
/*

	GLubyte Indices[] = {

		0,1,2,3,
		0,5,4,3,
		5,6,7,4,
		6,1,2,7,
		2,7,4,3,
		1,6,5,0
		};

*/

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
	glBufferData(GL_ARRAY_BUFFER, BufferSize, nodes, GL_STATIC_DRAW);

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

void CreateVBO2(float *nodes,GLshort *elem)
{
	/*Vertex Vertices[] =
	{
		{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 0

		// Top
		{ { -0.2f, 0.8f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } }, // 1
		{ { 0.2f, 0.8f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }, // 2
		{ { 0.0f, 0.8f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f } }, //3
		{ { 0.0f, 1.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },	// 4

		// Bottom
		{ { -0.2f, -0.8f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }, // 5
		{ { 0.2f, -0.8f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } }, // 6
		{ { 0.0f, -0.8f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f } }, //7
		{ { 0.0f, -1.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },	// 8

		// Left
		{ { -0.8f, -0.2f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } }, // 9
		{ { -0.8f, 0.2f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }, // 10
		{ { -0.8f, 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f } }, //11
		{ { -1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },	// 12

		// Right
		{ { 0.8f, -0.2f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }, // 13
		{ { 0.8f, 0.2f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } }, // 14
		{ { 0.8f, 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f } }, //15
		{ { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } }	// 16
	};

	GLubyte Indices[] = {
		// Top
		0, 1, 3,
		0, 3, 2,
		3, 1, 4,
		3, 4, 2,

		// Bottom
		0, 5, 7,
		0, 7, 6,
		7, 5, 8,
		7, 8, 6,

		// Left
		0, 9, 11,
		0, 11, 10,
		11, 9, 12,
		11, 12, 10,

		// Right
		0, 13, 15,
		0, 15, 14,
		15, 13, 16,
		15, 16, 14
	};

	GLubyte AlternateIndices[] = {
		// Outer square border:
		3, 4, 16,
		3, 15, 16,
		15, 16, 8,
		15, 7, 8,
		7, 8, 12,
		7, 11, 12,
		11, 12, 4,
		11, 3, 4,

		// Inner square
		0, 11, 3,
		0, 3, 15,
		0, 15, 7,
		0, 7, 11
	};
	*/
	GLenum ErrorCheckValue = glGetError();
	const size_t BufferSize = sizeof(nodes)*sizeof(float)*3;
	const size_t VertexSize = sizeof(float)*3; // Not sure
	const size_t RgbOffset = 0;
	
	glGenVertexArrays(1, &VaoId);
	glBindVertexArray(VaoId);
	
	glGenBuffers(1, &BufferId);
	glBindBuffer(GL_ARRAY_BUFFER, BufferId);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, nodes, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, VertexSize, 0);
//	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset);
	//glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, VertexSize, 0); // not sure

	glEnableVertexAttribArray(0);
//	glEnableVertexAttribArray(1);

	glGenBuffers(2, IndexBufferId);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[0]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elem)*sizeof(GLshort), elem, GL_STATIC_DRAW);

	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[1]);
	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(AlternateIndices), AlternateIndices, GL_STATIC_DRAW);

//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[0]); 

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

void renderOld(void) {

	angle+=deltaAngle;
	angleX+=deltaAngleX;

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
	glutSwapBuffers();
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
	if (state == GLUT_UP)
		deltaMove = 0.0f;	
	else{
	if(button==3){
		deltaMove = 3.0f;
		x += deltaMove * lx * 0.1f;
		z += deltaMove * lz * 0.1f;
		scale=1.2;
	}
	else if(button==4){
		deltaMove = -3.0f;
		x += deltaMove * lx * 0.1f;
		z += deltaMove * lz * 0.1f;
		scale=0.8;
	}
	   }

	//else{  // normal button event
	//       printf("Button %s At %d %d\n", (state == GLUT_DOWN) ? "Down" : "Up", x, y);
	   }

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


int main(int argc, char **argv) {
	
	//cout<<"started";
	// init GLUT and create window
	glutInit(&argc, argv);
//	glutInitContextVersion(3, 3);
	//glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
	//glutInitContextProfile(GLUT_CORE_PROFILE);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(640,480);
	glutCreateWindow("Test_TetgenModelRenderer_OpenGL");

	// register callbacks
	//glutReshapeFunc(ResizeFunction);
	glutDisplayFunc(RenderFunction);
//	glutDisplayFunc(renderOld);
	glutReshapeFunc(changeSize);
//	glutIdleFunc(RenderFunction);
	glutIdleFunc(IdleFunction);


	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);
	glutIgnoreKeyRepeat(1);
	glutSpecialUpFunc(releaseKey);

	glutCloseFunc(Cleanup);


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

	//CreateShaders();
//	CreateVBO();
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
 	//glInfo glInfo;
    	//glInfo.getInfo();




// Node File
	ifstream node_file("bar3.1.node");
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
		cout<<"\n";
		line_count++;
	}


// Elements File
	ifstream elem_file("bar3.1.ele");
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

cout<<"Nodes: "<<node_count<<"\n";
	cout<<"Dimensions: "<<node_dimensions<<"\n";
	cout<<"Elements: "<<elem_count<<"\n";
	cout<<"Nodes per Element: "<<elem_nodes<<"\n";

	CreateVBO();
//	CreateShaders();
	//CreateVBO2(nodes,elem);

	glutMainLoop();

	return 1;
}
