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
#include <iomanip>


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

using namespace std;


// Arrays from files
float *nodes;
float *nodes_orig;				// For serial code
GLshort *elem;
float *eigenVals;
float *eigenVecs;
int *fixed_nodes;
float *d_nodes;
float *Psy;


// Declare Coefficients 
// Square matrices dimensions: #eigenvals
float *d,*alpha, *alphaI,*beta,*gama, *C, *M;
float *F, *Fo, *q, *qo, *qd, *qdo, *u, *R, *Ro;


const int block_size=16;	// Change this according to NVIDIA Card

// Timestep
float h;


// General counters
int elem_count, elem_nodes;
int node_count, node_dimensions;
int eigencount;
unsigned int fixed_nodes_count;


// ToRenderOrNotToRender?, thats the question
bool render=false;

// Old coloring
float red=1.0f, blue=0.0f, green=0.0f;


// OpenGL rotating variables
float angle = 0.0f;
float deltaAngle = 0.0f;
int xOrigin = -1;

float angleX = 0.0f;
float deltaAngleX = 0.0f;
int yOrigin=-1;


// OpenGL Zoom
float zoom=-5.0;


// OpenGL "Dragging"
float posX=0.0, posY =0.0;
float DposX=0.0,DposY=0.0;
float DesposX=0.0, DesposY=0.0;
int buttonn=0;

// Force vector vars
float force=0.04;
bool force_changed=false;
float force_constant=0.01;
int force_axis=1;				// Y axis

// CUDA resources for buffer on GPU
cudaGraphicsResource *resources[1];
size_t size_resources;
float *cuda_dat = NULL;			// Pointer to map resources

bool CUDAResult;

// GLUT vars
int CurrentWidth = 800,
	CurrentHeight = 600,
	WindowHandle = 0;

// Parallel or Serial
bool parallel=false;

// Performance measurement
static unsigned int fps_start = 0;
static unsigned int fps_frames = 0;
int tpf;


// Virtual Buffer Objects (VBO's) vars
size_t BufferSize,VertexSize;
GLuint
    VertexShaderId,
	FragmentShaderId,
	ProgramId,
	VaoId,
	BufferId,
	IndexBufferId[2],
	ActiveIndexBuffer = 0;

// GLSL variables
//GLuint program;
GLint uniform_mvp;
GLint attribute_coord3d, attribute_v_color;


// Functions
void ResizeFunction(int Width, int Height);
void IdleFunction(void);
void RenderFunction(void);
void CreateVBO_CUDA(void);
void processSpecialKeys(int key, int x, int y);
void releaseKey(int key, int x, int y) ;
void mouseMove(int x, int y);
void mouseButton(int button, int state, int x, int y) ;
void change_force(float *F, float *Fo, unsigned int node_count, unsigned int fixed_nodes_count, unsigned int node_dimensions, float force, int force_axis);
void Cleanup(void);
void DestroyShaders(void);
void DestroyVBO(void);


// Functions to call CUDA (compiled with nvcc)
extern "C" void map_Texture(void *cuda_data, size_t size,cudaGraphicsResource *resource);
extern "C" bool displacement (float *h_q, float *h_qo, float *h_qd, float *h_qdo, float *h_F, float *h_Fo, float *h_Ro, float *h_alpha, float * h_alphaI, float *h_beta, float *h_gama, float *h_eigenVecs, float h_h, float *h_u, unsigned int eigencount, unsigned int node_count, unsigned int node_dimensions, const int block_size, float *buffer, float *h_nodes, int *fixed_nodes, unsigned int fixed_nodes_count, float *d_nodes, float *h_Psy);
extern "C" void* allocate_GPUnodes(float *nodes, unsigned int node_count, unsigned int node_dimensions);
extern "C" bool free_GPUnodes(float *d_nodes);



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
//	gluPerspective(45.0f, ratio, 0.1f, 100.0f); // DANGER
	glMatrixMode(GL_MODELVIEW);
}

void IdleFunction(void)
{

	fps_frames++;
    int delta_t = glutGet(GLUT_ELAPSED_TIME) - fps_start;
    if (delta_t > 1000) {
		// cout << delta_t / fps_frames << endl;
		tpf=delta_t / fps_frames;
		fps_frames = 0;
		fps_start = glutGet(GLUT_ELAPSED_TIME);
	}

	// Only calculate if needed
	if (force_changed=true){
		force_changed=false;
		change_force(F, Fo, node_count, fixed_nodes_count, node_dimensions, force, force_axis);
	}

	// Window Title Info
	char buffer[50];
	char axis;
	switch(force_axis){
	case 1:
		axis='Y';
		break;
	case 2:
		axis='X';
		break;
	case 3:
		axis='Z';
		break;

	}
	int n=sprintf(buffer,"TPF:%d, Force=%0.3f, Axis:%c, Render:%d",tpf,force,axis,render);
	glutSetWindowTitle(buffer);

	// Dragging and Rotating
	posX+=DposX;
	posY+=DposY;
	angle+=deltaAngle;
	angleX+=deltaAngleX;


	// GLM Matrices
	glm::vec3 axis_y(0, 1, 0);
	glm::vec3 axis_x(1, 0, 0);
	glm::mat4 anim = glm::rotate(glm::mat4(1.0f), angle, axis_y);
	glm::mat4 animy = glm::rotate(glm::mat4(1.0f), angleX, axis_x);

	// Push object so its not close to the camera
	glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, zoom));

//	glm::mat4 modelcenter = glm::translate(glm::mat4(1.0f), glm::vec3(0.4, 10.0, 0.0));

	// Lookat(eye,center,up) = position of cam, camera pointed to, top of the camera (tilted)
	glm::mat4 view = glm::lookAt(glm::vec3(0.0, 0.0, 0.0), glm::vec3(posX, posY, -5.0), glm::vec3(0.0, 1.0, 0.0));
	// Perspective
	glm::mat4 projection = glm::perspective(45.0f, 1.0f*CurrentWidth/CurrentHeight, 0.1f, 100.0f);

	// Calculate result
	glm::mat4 mvp = projection * view * model * anim * animy;

	glUseProgram(ProgramId);
	glUniformMatrix4fv(uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
	glutPostRedisplay();
}

void RenderFunction(void)
{
	// For rotation
	angle+=deltaAngle;
	angleX+=deltaAngleX;
   
    

	// Calculate & Render Displacement

    /* Parallel Code*/

	if(render==true){
		if(parallel==true){
            // Get pointer on GPU
			if(cudaGraphicsResourceGetMappedPointer((void**)&cuda_dat, &size_resources, *resources) !=cudaSuccess)
				printf("Resource pointer mapping failed...\n");

			CUDAResult = displacement (q, qo, qd, qdo, F, Fo, Ro, alpha, alphaI, beta, gama, eigenVecs, h, u, eigencount, node_count, node_dimensions, block_size, cuda_dat, nodes, fixed_nodes, fixed_nodes_count, d_nodes, Psy);
		}
		else{
    /*Serial Code*/

			displacement_serial(q, qo,qd, qdo, F, Fo, R, Ro, alpha, alphaI, beta, gama, eigenVecs, u, h, eigencount, node_count-fixed_nodes_count, node_dimensions, node_count, fixed_nodes, nodes, nodes_orig);
		
			glBindBuffer(GL_ARRAY_BUFFER, BufferId);
			glBufferSubData(GL_ARRAY_BUFFER,0, BufferSize, nodes);   
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
			glEnableVertexAttribArray(0); 
			
		}

		// Copy Old values
		std::copy(qd,qd+eigencount,qdo);
		std::copy(q,q+eigencount,qo);
//	std::copy(R,R+tot_rowCount,Ro);
		std::copy(F,F+((node_count-fixed_nodes_count)*node_dimensions),Fo);
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



	// OpenGL Render

	// Old Coloring
    // glColor3f(red,green,blue);
    // glDrawElements(GL_QUAD_STRIP, elem_count*elem_nodes*sizeof(elem[0]), GL_UNSIGNED_SHORT, NULL);
	glDrawElements(GL_TRIANGLE_STRIP, elem_count*elem_nodes, GL_UNSIGNED_SHORT, NULL);
    //glDrawElements(GL_TRIANGLE_STRIP, sizeBuffer/sizeof(GLshort), GL_UNSIGNED_SHORT, NULL);

	// glColor3f(1.0f,1.0f,1.0f);
    //glDrawElements(GL_LINE_LOOP, elem_count*elem_nodes*sizeof(elem[0]), GL_UNSIGNED_SHORT, NULL);
	glDrawElements(GL_LINES, elem_count*elem_nodes, GL_UNSIGNED_SHORT, NULL);
    //glDrawElements(GL_LINES, sizeBuffer/sizeof(GLshort), GL_UNSIGNED_SHORT, NULL);
    
	glutSwapBuffers();
	glutPostRedisplay();
}

void CreateVBO_CUDA(void)
{

    // Sizes
	BufferSize = sizeof(float)*node_count*node_dimensions; // 544: 32 per element on sphere
	VertexSize = sizeof(nodes[0])*node_dimensions; // Square: 12: 4 bytes (float) * node_dimension (3) on sphere
	
	// Declare VertexA
	glGenVertexArrays(1, &VaoId);
	glBindVertexArray(VaoId);
	
	// Nodes Buffer
	glGenBuffers(1, &BufferId);
	glBindBuffer(GL_ARRAY_BUFFER, BufferId);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, nodes, GL_DYNAMIC_DRAW);

	// Atributes
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);


 	// Element Buffer
	glGenBuffers(2, IndexBufferId);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[0]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, elem_count*elem_nodes*sizeof(elem),elem, GL_STATIC_DRAW); // Check Size

    // Register Pixel Buffer Object as CUDA graphics resource
	cudaGraphicsGLRegisterBuffer(resources, BufferId, cudaGraphicsMapFlagsNone);

    //Map the graphics resource
	if (cudaGraphicsMapResources(1, resources,0) != cudaSuccess)
        printf("Resource mapping failed...\n");
}

void CreateVBO(void)
{

    // Sizes
	BufferSize = sizeof(float)*node_count*node_dimensions; // 544: 32 per element on sphere
	VertexSize = sizeof(nodes[0])*node_dimensions; // Square: 12: 4 bytes (float) * node_dimension (3) on sphere
	
	// Declare VertexA
	glGenVertexArrays(1, &VaoId);
	glBindVertexArray(VaoId);
	
	// Nodes Buffer
	glGenBuffers(1, &BufferId);
	glBindBuffer(GL_ARRAY_BUFFER, BufferId);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, nodes, GL_DYNAMIC_DRAW);

	// Atributes
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);


 	// Element Buffer
	glGenBuffers(2, IndexBufferId);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[0]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, elem_count*elem_nodes*sizeof(elem),elem, GL_STATIC_DRAW); // Check Size

    // Register Pixel Buffer Object as CUDA graphics resource
//	cudaGraphicsGLRegisterBuffer(resources, BufferId, cudaGraphicsMapFlagsNone);

    //Map the graphics resource
//	if (cudaGraphicsMapResources(1, resources,0) != cudaSuccess)
	//      printf("Resource mapping failed...\n");
}

void Cleanup(void){
	DestroyVBO();
	DestroyShaders();
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

	// Unmap CUDA Resources
    cudaGraphicsUnmapResources(1, resources);
}

 void CreateShaders(void)
 {

     // Shaders / GLSL

	 GLenum ErrorCheckValue = glGetError();

	 GLint link_ok = GL_FALSE;

	 if ((VertexShaderId = create_shader("mwgpu.v.glsl", GL_VERTEX_SHADER)) == 0)
		 exit(-1);
	 if ((FragmentShaderId = create_shader("mwgpu.f.glsl", GL_FRAGMENT_SHADER)) == 0)
		 exit(-1);


	 ProgramId= glCreateProgram();
	 glAttachShader(ProgramId, VertexShaderId);
	 glAttachShader(ProgramId, FragmentShaderId);
	 glLinkProgram(ProgramId);
	 glGetProgramiv(ProgramId, GL_LINK_STATUS, &link_ok);
	 if (!link_ok) {
		 fprintf(stderr, "glLinkProgram:");
		 print_log(ProgramId);
	 }

	 const char* attribute_name;
	 attribute_name = "coord3d";
	 attribute_coord3d = glGetAttribLocation(ProgramId, attribute_name);
	 if (attribute_coord3d == -1) {
		 fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
	 }
	 attribute_name = "v_color";
	 attribute_v_color = glGetAttribLocation(ProgramId, attribute_name);
	 attribute_v_color = (1.0f,1.0f,1.0f);
	 if (attribute_v_color == -1) {
		 fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
	 }
	 const char* uniform_name;
	 uniform_name = "mvp";
	 uniform_mvp = glGetUniformLocation(ProgramId, uniform_name);
	 if (uniform_mvp == -1) {
		 fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
	 }

	 ErrorCheckValue = glGetError();
	 if (ErrorCheckValue != GL_NO_ERROR)
	 {
		 fprintf(
			 stderr,
			 "ERROR: Could not create a shader: %s \n",
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

	if (key == 27)				// Escape
		exit(0);
	if (key==32){				// Space
		if(render==true)
			render=false;
		else
			render=true;
	}
}



void processSpecialKeys(int key, int x, int y) 
{

	switch(key) {
	case GLUT_KEY_F1 :
		force-=force_constant;
		force_changed=true;
		break;
		// red = 1.0;
		// green = 0.0;
		// blue = 0.0;
	case GLUT_KEY_F2 :
		force+=force_constant;
		force_changed=true;
		break;
		// red = 0.0;
		// green = 1.0;
		// blue = 0.0;
	case GLUT_KEY_F3 :
		force_changed=true;
		force_axis=1;
		break;
		// red = 0.0;
		// green = 0.0;
		// blue = 1.0; 
	case GLUT_KEY_F4 :
		force_changed=true;
		force_axis=2;
		break;
	case GLUT_KEY_F5 :
		force_changed=true;
		force_axis=3;
		break;
		// red = 1.0;
		// green = 1.0;
		// blue = 1.0;

		// Rotate object with keys
		case GLUT_KEY_LEFT : deltaAngle = -0.35f; break;
		case GLUT_KEY_RIGHT : deltaAngle = 0.35f; break;
		case GLUT_KEY_UP : deltaAngleX = -0.35f; break;
		case GLUT_KEY_DOWN : deltaAngleX = 0.35f; break;
	}
}

void releaseKey(int key, int x, int y) 
{
	// Reset Values
	switch (key) {
		case GLUT_KEY_LEFT :
		case GLUT_KEY_RIGHT : deltaAngle = 0.0f;break;
		case GLUT_KEY_UP :
		case GLUT_KEY_DOWN : deltaAngleX = 0.0f;break;
	}
}

void mouseMove(int x, int y)
{ 	
	// Left button is down
	if (buttonn > 0) {

		// update deltaAngle
		deltaAngle = (x - xOrigin) * 0.01f;
		deltaAngleX = (y - yOrigin) * 0.01f;
	}
	else if (buttonn <0){

		DesposX=x;
		DesposY=y;
		DposX= -(x-xOrigin);
		DposY= (y-yOrigin);
		DposX*=0.0005;
		DposY*=0.0005;

	}
}

void mouseButton(int button, int state, int x, int y) 
{

	// 3 == scroll up, 4 == scroll down
	if ((button == 3) || (button == 4))
	{
		if(button==3){
			zoom*=0.95;
		}
		else if(button==4){
			zoom*=1.05;
		}
	}

	if (button == GLUT_LEFT_BUTTON) {

		// when the button is released
		if (state == GLUT_UP) {
			deltaAngle=0;
			deltaAngleX=0;
			xOrigin = -1;
			yOrigin=-1;
			buttonn=0;
		}
		else  {
			buttonn=1;
			xOrigin = x;
			yOrigin= y;
		}
	}
	if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_UP) {	
			buttonn=0;
			DposX=0;
			DposY=0;
			xOrigin = -1;
			yOrigin=-1;
		}
		else  {
			buttonn=-1;
			xOrigin = x;
			yOrigin= y;
		}
	}
}

void change_force(float *F, float *Fo, unsigned int node_count, unsigned int fixed_nodes_count, unsigned int node_dimensions, float force, int force_axis)
{

	// Delcare F in only 1 direction
	for (int i=0;i<(node_count-fixed_nodes_count)*node_dimensions;i++){
		if(3-(i%3)==force_axis)
			F[i]=force;
		else
			F[i]=0;
	}
	// Copy F to Fo
	std::copy(F,F+((node_count-fixed_nodes_count)*node_dimensions),Fo);
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

	glClearColor(0.25f, 0.25f, 0.25f, 0.0f);
	// glClearColor(1.0f, 1.0f, 1.0f, 0.0f);


/*#########---------Ends OpenGL Init------------#########*/

/*#########---------Starts Getopts code------------#########*/


// Extern because vars declared into getopts
	extern char *optarg;
	extern int optind;
	int c, err = 0; 
	int nflag=0, kflag=0;
	string ename,nname,kname,eigenvec_name, fixedname, psyname;
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
			psyname.append(optarg);
			psyname.append("_Psy.csv");
			break;

			// Not used atm
		// case 'k':
		// 	kflag = 1;
		// 	kname.append(optarg);
		// 	kname.append(".csv");
		// 	break;
		// case '?':
		// 	err = 1;
		// 	break;
		}
	if (nflag == 0) 
	{	/* -n (Node_filename is mandatory */
		fprintf(stderr, "%s: Node_filename is missing \n", argv[0]);
		fprintf(stderr, usage, argv[0]);
		exit(1);
	} 

	// Not used atm
	// else if (kflag==0)
	// {
	// 	// TODO fix nmname to get rid of extension
	// 	//	kname=nname;
	// }

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
	// Testing printf
    printf("EigenFile:%s\n", kname.c_str());
	cout<<"EigenVecsFile:"<<eigenvec_name<<"\n";

	if (optind < argc)
	{	/* Last arguments */
		for (; optind < argc; optind++)
			printf("argument: \"%s\"\n", argv[optind]);
//	else {
//		printf("no arguments left to process\n");
	}

/*#########---------End of Getopts------------#########*/

/*#########---------File Loading------------#########*/


/* Open Files  */

	// Node File
	ifstream node_file(nname.c_str());
	string line;
	
	int line_count=0;
	int node_start=0;

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
				if(line_count==0)
				{
					if(val_count==0)
						// Get Number of Nodes
						node_count=atoi(cell.c_str());
					else if(val_count==1){
						//Get Number of Dimensions
						node_dimensions=atoi(cell.c_str());
						nodes=new float[node_count*node_dimensions];
						nodes_orig=new float[node_count*node_dimensions];

					}

				}
				else
				{
					// Check speed of conversion
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
							nodes[node_dimensions*(line_count-1)+(val_count-1)]=converted;
					}
				}
				val_count++;
			}
		}
		line_count++;
	}


    // Elements File
	ifstream elem_file(ename.c_str());	
	line_count=0;
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
				if(line_count==0)
				{
					if(val_count==0)
						// Get Number of Elements
						elem_count=atoi(cell.c_str());
					else if(val_count==1){
						//Get Number of Element Nodes
						elem_nodes=atoi(cell.c_str());
						// Allocate array
						elem=new GLshort[elem_count*elem_nodes];
					}
				}
				else
				{
					char* p;
					GLshort converted = strtol(cell.c_str(), &p,10); // Add base 10
					if (*p) 
					{
						// conversion failed because the input wasn't a number
					}
					else 
					{
						if(val_count>0)
						{
							elem[elem_nodes*(line_count-1)+(val_count-1)]=converted-node_start;
						}
					}
				}
				val_count++;
			}
		}
		line_count++;
	}

	// EigenFile

	// No need for columns in EIGENVALUES
    // Get file handle
	std::ifstream csv_file_rows(kname.c_str());
	std::ifstream csv_file(kname.c_str());
	unsigned int row_count=0;
	unsigned int tot_rowCount;

    // Read whole file to get row and count - Don't like it but its only done once
	while (std::getline(csv_file_rows,line)){
		row_count++;
	}

	tot_rowCount=row_count;

	// Allocate array
	eigenVals=new float[tot_rowCount];

	row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(csv_file,line)){
		// Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;
		
		// Separate values by ',' (CSV)
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also

				// Convert char* to float
				char *p;
				float converted = strtof(colVal.c_str(),&p);
				if (*p){}
				else{
					eigenVals[row_count]=converted;
				}
			}
		}
		row_count++;
	}


    // Fixed Nodes

	// No need for columns in Fixed Nodes either
    // Get file handle
	std::ifstream fixed_nodes_rows(fixedname.c_str());
	std::ifstream fixed_nodes_file(fixedname.c_str());

    row_count=0;				// Reset

    // Read whole file to get row and count
	while (std::getline(fixed_nodes_rows,line)){
		row_count++;
	}

	fixed_nodes_count=row_count;

	// Allocate array
	fixed_nodes=new int[fixed_nodes_count];

	row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(fixed_nodes_file,line)){

		// Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;

		// Separate values by ',' (CSV)
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also
				// Convert char* to float
				char *p;
				int converted = strtol(colVal.c_str(),&p,10);
				if (*p){}
				else{
					fixed_nodes[row_count]=converted-1;
				}
			}
		}
		row_count++;
	}

    // EigenVectorsFile

    // Get file handle

	std::ifstream eigenvec_file_rows(eigenvec_name.c_str());
	std::ifstream eigenvec_file(eigenvec_name.c_str());

	// Columns should be same as eigenVals count
	// Rows should be the same as nodes x dimensions
	unsigned int col_countEvec=0;
	unsigned int tot_colCountEvec;
	
	// Columns same as EigenVals count
	tot_colCountEvec=tot_rowCount;

	// Allocate array
	eigenVecs=new float[(node_count-fixed_nodes_count)*node_dimensions*tot_colCountEvec];

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

		// Separate values by ',' (CSV)
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also
				// Convert char* to float
				char *p;
				float converted = strtof(colVal.c_str(),&p);
				if (*p){}
				else{
					eigenVecs[(tot_colCountEvec*row_count)+col_countEvec]=converted;
				}
			}
			col_countEvec++;
		}
		row_count++;
	}

	// Psy
    // Get file handle

	std::ifstream psy_file(psyname.c_str());

	// Columns should be same as eigenVals count
	// Rows should be the same as nodes x dimensions
    col_countEvec=0;			// Reuse from EigenVecs	


	// Allocate array
	Psy=new float[(node_count-fixed_nodes_count)*node_dimensions*tot_colCountEvec];

	row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(psy_file,line)){

		// Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;
		
		// Reset col value
		col_countEvec=0;

		// Separate values by ',' (CSV)
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also
				// Convert char* to float
				char *p;
				float converted = strtof(colVal.c_str(),&p);
				if (*p){}
				else{
					Psy[(tot_colCountEvec*row_count)+col_countEvec]=converted;
				}
			}
			col_countEvec++;
		}
		row_count++;
	}


/*#########---------Ends File Loading---------------#########*/


/*#########---------Allocate coefficient matrices---------------#########*/

	// D, alpha, beta, gamma square matrices eigencount x eigencount

	// Timestep
	h=0.03;

	d = new float[tot_rowCount*tot_rowCount];
	alpha = new float[tot_rowCount*tot_rowCount];
	beta = new float[tot_rowCount*tot_rowCount];
	gama = new float[tot_rowCount*tot_rowCount];
	M =  new float[tot_rowCount*tot_rowCount];
	C =  new float[tot_rowCount];

	// Create alpha-identity
	alphaI = new float[tot_rowCount*tot_rowCount];


	// 2 nested fors to declare square diagonal matrices
	for(int i=0;i<tot_rowCount;i++){
		// Maybe only need 1 for ? - TODO
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

	change_force(F, Fo, node_count, fixed_nodes_count, node_dimensions, force, force_axis);
	
	// For loop to initialize all other vectors (all 0's)
	for (int i=0;i<tot_rowCount;i++){
		q[i]=0;
		qo[i]=0;
		qd[i]=0;
		qdo[i]=0;
	}


/*#########---------Ends calculation vectors allocation---------------#########*/

	if(parallel==true)
		CreateVBO_CUDA();
	else
		CreateVBO();
	CreateShaders();

	eigencount=tot_rowCount;

	// Print some info
	cout<<"Nodes: "<<node_count<<"\t";
	cout<<"Dimensions: "<<node_dimensions<<"\n";
	cout<<"Elements: "<<elem_count<<"\t";
	cout<<"Nodes per Element: "<<elem_nodes<<"\n";
	std::cout<<"# Eigenvalues: "<<tot_rowCount<<"\n";
	std::cout<<"Eigenvectors matrix dimensions:"<<node_count*node_dimensions<<"x"<<tot_rowCount<<"\n";


	std::copy(nodes,nodes+node_count*node_dimensions,nodes_orig);

	// Allocate GPU globals before main loop instead of doing it every time
	allocate_GPUnodes( nodes, node_count, node_dimensions);


	fps_start = glutGet(GLUT_ELAPSED_TIME);

	glutMainLoop();


	free_GPUnodes(d_nodes);		// TODO - pass device globals

	cudaDeviceReset();


/*#########---------Free System Memory---------------#########*/


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
	free(nodes_orig);

/*#########---------Free Memory---------------#########*/

	return 1;

}
