#include "mwgpu.h"

using namespace std;


// Arrays from files
float *nodes;
float *nodes_orig;				// For serial code
GLushort *elem;
float *eigenVals;
float *eigenVecs;
int *fixed_nodes;
float *Psi;


// Declare Coefficients 
// Square matrices dimensions: #eigenvals
float *d,*alpha, *alphaI,*beta,*gama, *C, *M;
float *F, *Fo, *q, *qo, *qd, *qdo, *u, *R, *Ro;

const int block_size=16;	// Change this according to NVIDIA Card
unsigned int maxThreadsBlock = 512;

// Timestep
float h;

// General counters & structs
int eigencount;
nodes_struct ns;
elem_struct es;

// Tetgen Files might have an offset
int node_init_index;

// ToRenderOrNotToRender?, thats the question
// bool render=false;
bool render=false;

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
int force_axis=1;				// Default to Y axis

// CUDA resources for buffer on GPU
cudaGraphicsResource *resources[1];
size_t size_resources;
float *cuda_dat = NULL;			// Pointer to map resources

bool CUDAResult;				// No Error on CUDA functions

// GLUT vars
int CurrentWidth = 800,
	CurrentHeight = 600,
	WindowHandle = 0;

// Parallel or Serial
bool parallel=true;

// Render only Lines
bool onlyNodes=false;

// Performance measurement
static unsigned int fps_start = 0;
static unsigned int fps_frames = 0;
int tpf;						// Time Per Frame in msec
float simtime;					// Time measurement of Simulation in msec

// Debug/Verbose mode
bool debug=false;
std::string logname="log";
vector<string> lines;


// Virtual Buffer Objects (VBO's) vars
size_t BufferSize,VertexSize;
GLuint
    VertexShaderId,
    VertexShaderId2,
	FragmentShaderId,
	FragmentShaderLinesId,
	ProgramId,
	ProgramId2,
	VaoId,
	BufferId,
	IndexBufferId,
	ActiveIndexBuffer = 0;

// GLSL variables
GLint uniform_mvp;
GLint attribute_coord3d, attribute_v_color;

// ObjFiles
string obj_Filename;

void GLM_MVP(GLuint pId, nodes_struct ns){
// GLM Matrices
	glm::vec3 axis_y(0, 1, 0);
	glm::vec3 axis_x(1, 0, 0);
	glm::mat4 anim = glm::rotate(glm::mat4(1.0f), angle, axis_y);
	glm::mat4 animy = glm::rotate(glm::mat4(1.0f), angleX, axis_x);

	// Push object so its not close to the camera
	// Should rotate around its own center
	glm::mat4 model = glm::translate(glm::mat4(1.0f),glm::vec3(ns.center_x,ns.center_y,ns.center_z+zoom));
	glm::mat4 model2 = glm::translate(glm::mat4(1.0f),glm::vec3(-ns.center_x,-ns.center_y,-ns.center_z));

	// Lookat(eye,center,up) = position of cam, camera pointed to, top of the camera (tilted)
	glm::mat4 view = glm::lookAt(glm::vec3(ns.center_x,ns.center_y, ns.center_z), glm::vec3(ns.center_x+posX, ns.center_y+posY, ns.center_z -3.0), glm::vec3(0.0, 1.0, 0.0));
	// Perspective
	glm::mat4 projection = glm::perspective(45.0f, 1.0f*CurrentWidth/CurrentHeight, 0.1f, 1000.0f);

	// Calculate result
	glm::mat4 mvp = projection * view * model * anim * animy * model2;

	glUseProgram(pId);
	// glUseProgram(ProgramId2);
	glUniformMatrix4fv(uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));

	glutPostRedisplay();

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
	glMatrixMode(GL_MODELVIEW);
}

void IdleFunction(void)
{
   	fps_frames++;
    int delta_t = glutGet(GLUT_ELAPSED_TIME) - fps_start;
    if (delta_t > 1000) {
		tpf=delta_t / fps_frames;
		fps_frames = 0;
		fps_start = glutGet(GLUT_ELAPSED_TIME);
	}

	// Only calculate if needed
	if (force_changed=true){
		force_changed=false;
		change_force(F, Fo, ns, force, force_axis);
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


 // Calculate & Render Displacement

    /* Parallel Code*/

	if(render==true){
		if(parallel==true){
            // Get pointer on GPU
			if(cudaGraphicsResourceGetMappedPointer((void**)&cuda_dat, &size_resources, *resources) !=cudaSuccess)
				printf("Resource pointer mapping failed...\n");

			CUDAResult = displacement (Fo, h, eigencount, ns.count, ns.dimensions, &simtime,block_size, cuda_dat, fixed_nodes, ns.fixed_count, maxThreadsBlock, debug);
		}
		else{
    /*Serial Code*/
	        // Time simulation
			timespec before, after, tv;
			clock_gettime(CLOCK_MONOTONIC, &before); // Get time before simulation
			
			displacement_serial(q, qo,qd, qdo, F, Fo, R, Ro, alpha, alphaI, beta, gama, eigenVecs, u, h, eigencount, ns.count-ns.fixed_count, ns.dimensions, ns.count, fixed_nodes, nodes, Psi, nodes_orig);
		
			glBindBuffer(GL_ARRAY_BUFFER, BufferId);
			glBufferSubData(GL_ARRAY_BUFFER,0, BufferSize, nodes);   
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
			glEnableVertexAttribArray(0); 
			
			// Copy Old values
			std::copy(qd,qd+eigencount,qdo);
			std::copy(q,q+eigencount,qo);
			clock_gettime(CLOCK_MONOTONIC, &after);
			tv=diff_time(before,after);
			simtime=tv.tv_nsec=tv.tv_nsec; // Get time in miliseconds
			simtime/=1000000;
		}
   		std::copy(F,F+((ns.count-ns.fixed_count)*ns.dimensions),Fo);
	}
	
	int n=sprintf(buffer,"TPF:%d, SimTime:%0.3fms, Force=%0.3f, Axis:%c, Render:%d, Parallel:%d",tpf,simtime,force,axis,render,parallel);

	// Set new title
	glutSetWindowTitle(buffer);

	// Dragging and Rotating
	posX+=DposX;
	posY+=DposY;
	angle+=deltaAngle;
	angleX+=deltaAngleX;
}

void RenderFunction(void)
{
	// For rotation
	angle+=deltaAngle;
	angleX+=deltaAngleX;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	if(onlyNodes==false){
		GLM_MVP(ProgramId, ns);
		glDrawElements(GL_QUADS, es.count*es.nodes, GL_UNSIGNED_SHORT, NULL);
		GLM_MVP(ProgramId2, ns);
		glDrawElements(GL_LINES, es.count*es.nodes, GL_UNSIGNED_SHORT, NULL);
	}
	else{
		GLM_MVP(ProgramId2, ns);
		glPointSize(2.8f);
		glDrawElements(GL_POINTS, es.count*es.nodes, GL_UNSIGNED_SHORT, NULL);
	}
    
	glutSwapBuffers();
	glutPostRedisplay();
}

void CreateVBO_CUDA(void)
{

    // Sizes
	BufferSize = sizeof(float)*ns.count*ns.dimensions;
	VertexSize = sizeof(nodes[0])*ns.dimensions;
	
	// Declare VertexA
	glGenVertexArrays(1, &VaoId);
	glBindVertexArray(VaoId);
	
	// Nodes Vertex Buffer Objects
	glGenBuffers(1, &BufferId);
	glBindBuffer(GL_ARRAY_BUFFER, BufferId);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, nodes, GL_DYNAMIC_DRAW);

	// Atributes
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);


 	// Element Buffer Objects
	glGenBuffers(1, &IndexBufferId);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId);
	// glBufferData(GL_ELEMENT_ARRAY_BUFFER, es.count*es.nodes*sizeof(elem),elem, GL_STATIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, es.count*es.nodes*sizeof(elem),NULL, GL_STATIC_READ);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,0,es.count*es.nodes*sizeof(elem)/2,elem);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,es.count*es.nodes*sizeof(elem)/2,es.count*es.nodes*sizeof(elem)/2,&elem[es.count*es.nodes/2]);

    // Register Pixel Buffer Object as CUDA graphics resource
	cudaGraphicsGLRegisterBuffer(resources, BufferId, cudaGraphicsMapFlagsNone);

    //Map the graphics resource
	if (cudaGraphicsMapResources(1, resources,0) != cudaSuccess)
        printf("Resource mapping failed...\n");
}

void CreateVBO(void)
{

    // Sizes
	BufferSize = sizeof(float)*ns.count*ns.dimensions;
	VertexSize = sizeof(nodes[0])*ns.dimensions;
	

	// Declare VertexA
	glGenVertexArrays(1, &VaoId);
	glBindVertexArray(VaoId);
	
	// Nodes Vertex Buffer Objects
	glGenBuffers(1, &BufferId);
	glBindBuffer(GL_ARRAY_BUFFER, BufferId);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, nodes, GL_DYNAMIC_DRAW);

	// Atributes
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

// Element Buffer Objects
	glGenBuffers(1, &IndexBufferId);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, es.count*es.nodes*sizeof(elem),NULL, GL_STATIC_READ);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,0,es.count*es.nodes*sizeof(elem)/2,elem);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,es.count*es.nodes*sizeof(elem)/2,es.count*es.nodes*sizeof(elem)/2,&elem[es.count*es.nodes/2]);
 	

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
	glDeleteBuffers(1, &IndexBufferId);

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

	// UnMap CUDA Resources
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
	 if ((VertexShaderId2 = create_shader("mwgpu.v.glsl", GL_VERTEX_SHADER)) == 0)
		 exit(-1);
	 if ((FragmentShaderLinesId = create_shader("mwgpu.fl.glsl", GL_FRAGMENT_SHADER)) == 0)
		 exit(-1);

	 ProgramId2= glCreateProgram();
	 glAttachShader(ProgramId2, VertexShaderId2);
	 glAttachShader(ProgramId2, FragmentShaderLinesId);
	 glLinkProgram(ProgramId2);

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
	glDetachShader(ProgramId2, VertexShaderId2);
	glDetachShader(ProgramId2, FragmentShaderLinesId);

	glDeleteShader(FragmentShaderId);
	glDeleteShader(VertexShaderId);
	glDeleteShader(VertexShaderId2);
	glDeleteShader(FragmentShaderLinesId);;

	glDeleteProgram(ProgramId);
	glDeleteProgram(ProgramId2);

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
		glutLeaveMainLoop();	// To be able to measure performance of CUDA NVP
	if (key==32){				// Space
		if(render==true)
			render=false;
		else
			render=true;
	}
	if (key==8){				// Backspace
		if(onlyNodes==true)
			onlyNodes=false;
		else
			onlyNodes=true;
	}
	if (key == 9)				// Tab
		save_Obj(obj_Filename, nodes, elem, ns.count, ns.dimensions, es.count, es.nodes, node_init_index);	// Save OBJ File
//		save_tetgen(obj_Filename, nodes, elem, ns.count, ns.dimensions, es.count, es.nodes);	// Save Node File

}



void processSpecialKeys(int key, int x, int y) 
{

	switch(key) {
	case GLUT_KEY_F1 :
		force-=force_constant;
		force_changed=true;
		break;
	case GLUT_KEY_F2 :
		force+=force_constant;
		force_changed=true;
		break;
	case GLUT_KEY_F3 :
		force_changed=true;
		force_axis=1;
		break;
	case GLUT_KEY_F4 :
		force_changed=true;
		force_axis=2;
		break;
	case GLUT_KEY_F5 :
		force_changed=true;
		force_axis=3;
		break;
	case GLUT_KEY_F6 :
		force_changed=true;
		force=-force;
		break;


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
			// std::cout<<x<<","<<y<<std::endl;
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

void change_force(float *F, float *Fo, nodes_struct ns, float force, int force_axis)
{

	// Delcare F in only 1 direction
	for (int i=0;i<(ns.count-ns.fixed_count)*ns.dimensions;i++){
		if(3-(i%3)==force_axis)
			F[i]=force;
		else
			F[i]=0;
	}
	// Copy F to Fo
	std::copy(F,F+((ns.count-ns.fixed_count)*ns.dimensions),Fo);
}




int main(int argc, char **argv)
{

/*#########---------OpenGL Init------------#########*/

	// Init GLUT & Create Window
	if(debug==true) std::cout<<"Initializing GLUT..."<<std::endl;	
	glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(640,480);
	glutCreateWindow("Modal_Warping_OpenGL_CUDA");

	// Register callbacks
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
	if(debug==true){
		fprintf(
			stdout,
			"INFO: OpenGL Version: %s\n",
			glGetString(GL_VERSION)
			);
	}

	// Background Color
	glClearColor(0.25f, 0.25f, 0.25f, 0.0f);
	// glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

	// Initialize Log File
	init_log(logname);


/*#########---------Ends OpenGL Init------------#########*/


/*#########---------Starts Getopts code------------#########*/


// Extern because vars declared into getopts
	extern char *optarg;
	extern int optind;
	int c, err = 0; 
	int nflag=0, kflag=0;
	string nname,ename,kname,eigenvec_name, fixedname, psiname;
	static char usage[] = "usage: %s -n Node_filename [-p 1 || 0] [-t threadsonGPU] \n";

	while ((c = getopt(argc, argv, "n:p:t:d:e:")) != -1)
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
			psiname.append(optarg);
			psiname.append("_Psi.csv");
			obj_Filename.append(optarg);
			break;
		case 'e':
			// Convert to strings & append extensions
			kname.assign(optarg);
			kname.append(".csv");
			eigenvec_name.assign(optarg);
			eigenvec_name.append("_vec.csv");
			psiname.assign(optarg);
			psiname.append("_Psi.csv");
		case 'p':
			istringstream(optarg) >> parallel;
			break;
        case 't':
			maxThreadsBlock=atoi(optarg);
			break;
		case 'd':
			istringstream(optarg) >> debug;
			break;
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

	if (optind < argc)
	{	/* Last arguments */
		for (; optind < argc; optind++)
			printf("argument: \"%s\"\n", argv[optind]);
//	else {
//		printf("no arguments left to process\n");
	}

/*#########---------End of Getopts------------#########*/

/*#########---------File Loading------------#########*/

	std::cout<<"Loading Files..."<<std::endl;


	if(debug==true) lines.push_back("Loading Node File...");
	node_init_index = load_nodefile(nname.c_str(),nodes, &ns);

	if(debug==true) lines.push_back("Loading Elem File...");	
	load_elemfile(ename.c_str(),elem, &es, node_init_index);

	if(debug==true) lines.push_back("Loading EigenVals File...");	
	eigencount = load_eigenvalsfile(kname.c_str(),eigenVals);

	if(debug==true) lines.push_back("Loading Fixed File...");	
	load_fxdnodesfile(fixedname.c_str(), fixed_nodes ,&ns);
	
	if(debug==true) lines.push_back("Loading EigenVecs File...");	
	load_eigenvecfile(eigenvec_name.c_str(), eigenVecs , &ns, eigencount);
	
	if(debug==true) lines.push_back("Loading Psi File...");	
	load_Psifile(psiname.c_str(), Psi , &ns, eigencount);
	
	std::cout<<"Done Loading Files."<<std::endl;	

	if(debug==true) log(logname,lines);

/*#########---------Ends File Loading---------------#########*/


/*#########---------Allocate coefficient matrices---------------#########*/

	// D, alpha, beta, gamma square matrices eigencount x eigencount

	// Timestep
	h=0.03;

	std::cout<<"Allocating Arrays..."<<std::endl;
	if(debug==true) {
		lines.push_back("Allocating Arrays...");
		log(logname,lines);
	}
	d = new float[eigencount*eigencount];
	alpha = new float[eigencount*eigencount];
	beta = new float[eigencount*eigencount];
	gama = new float[eigencount*eigencount];
	M =  new float[eigencount*eigencount];
	C =  new float[eigencount];

	// Create alpha-identity
	alphaI = new float[eigencount*eigencount];


	// 2 nested fors to declare square diagonal matrices
	for(int i=0;i<eigencount;i++){
		for(int j=0;j<eigencount;j++){
			int index=i*eigencount+j;
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
	F = new float[(ns.count-ns.fixed_count)*ns.dimensions];
	Fo = new float[(ns.count-ns.fixed_count)*ns.dimensions];
	q = new float[eigencount];
	qo = new float[eigencount];
	qd = new float[eigencount];
	qdo = new float[eigencount];
	u = new float[ns.count*ns.dimensions];;

	change_force(F, Fo, ns, force, force_axis);
	
	// For loop to initialize all other vectors (all 0's)
	for (int i=0;i<eigencount;i++){
		q[i]=0;
		qo[i]=0;
		qd[i]=0;
		qdo[i]=0;
	}
	std::cout<<"Done Allocating Arrays."<<std::endl;	

/*#########---------Ends calculation vectors allocation---------------#########*/

	std::cout<<"Creating Buffer Objects..."<<std::endl;	
	if(parallel==true)
		CreateVBO_CUDA();
	else
		CreateVBO();

	std::cout<<"Creating Shaders..."<<std::endl;	
	CreateShaders();	

	//if(debug==true){
		std::cout<<"Nodes: "<<ns.count<<"\t";
		std::cout<<"Dimensions: "<<ns.dimensions<<std::endl;
		std::cout<<"Elements: "<<es.count<<"\t";
		std::cout<<"Nodes per Element: "<<es.nodes<<std::endl;
		std::cout<<"# Eigenvalues: "<<eigencount<<std::endl;
		std::cout<<"Eigenvectors matrix dimensions: "<<ns.count*ns.dimensions<<"x"<<eigencount<<std::endl;
//	}

	// Save original node locations on a different variable
	nodes_orig=new float[ns.count*ns.dimensions];					
	std::copy(nodes,nodes+ns.count*ns.dimensions,nodes_orig);


	if(parallel==true){
		// Allocate GPU globals before main loop instead of doing it every time
		if(debug==true) std::cout<<"Allocating GPU globals..."<<std::endl;	
		allocate_GPUmem(nodes, alphaI, alpha, beta, gama, eigenVecs, Psi, ns.count, ns.dimensions, ns.fixed_count, eigencount);
	}

	// For OpenGL performance
	fps_start = glutGet(GLUT_ELAPSED_TIME);

	if(debug==true) std::cout<<"Looping on OpenGL..."<<std::endl;	
	glutMainLoop();

	if(parallel==true){
		if(debug==true) std::cout<<"Freeing up GPU memory..."<<std::endl;	
		// Free up memory
		free_GPUnodes();
		// Reset GPU
		cudaDeviceReset();
	}


/*#########---------Free System Memory---------------#########*/

	if(debug==true) std::cout<<"Freeing up system memory..."<<std::endl;	
	delete nodes;
	delete elem;
	delete eigenVals;
	delete eigenVecs;
	delete d;
	delete alpha;
	delete alphaI;
	delete beta;
	delete gama;
	delete M;
	delete C;
	delete F;
	delete Fo;
	delete q;
	delete qo;
	delete qd;
	delete qdo;
	delete u;
	delete nodes_orig;
	delete Psi;

/*#########---------Free Memory---------------#########*/

	return 0;

}
