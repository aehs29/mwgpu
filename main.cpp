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


#include <GL/glut.h>


/* CUDA Includes */
#include <cuda_runtime.h>
#include <vector_types.h>

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


	bool bTestResult;
extern "C" bool runTest(const int argc, const char **argv, float *nodes, int node_count);

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
}


void renderScene(void) {

    bTestResult = runTest(0, (const char **)"", nodes, node_count);


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

//	glBegin(GL_TRIANGLES);
//		glVertex3f(-2.0f,-2.0f, 0.0f);
//		glVertex3f( 2.0f, 0.0f, 0.0);
//		glVertex3f( 0.0f, 2.0f, 0.0);
//	glEnd();
	// ########------Tetrahedron ends-------#######
	glutSwapBuffers();
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
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(640,480);
	glutCreateWindow("Test_TetgenModelRenderer_OpenGL_CUDA");

	// register callbacks
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);

	// here are the new entries
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);
	glutIgnoreKeyRepeat(1);
	glutSpecialUpFunc(releaseKey);



	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);


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
	//int node_count;
//	int node_dimensions;
	//float **nodes;
	while(std::getline(node_file,line))
	{
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
	ifstream elem_file(ename.c_str());
	//string line;
	
	line_count=0;
	//int elem_count;
	//int elem_nodes;
	//int **elem;
	while(std::getline(elem_file,line))
	{
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
							elem[elem_nodes*(line_count-1)+(val_count-1)]=converted-1;
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
		cout<<"\n";
		line_count++;
	}

	

    // run the device part of the program
    bTestResult = runTest(argc, (const char **)argv, nodes, node_count);

	for (int ii=0;ii<node_count*3;ii++)
	{
		cout<<nodes[ii]<<"\n";
	}

	cout<<"Nodes: "<<node_count<<"\n";
	cout<<"Dimensions: "<<node_dimensions<<"\n";
	cout<<"Elements: "<<elem_count<<"\n";
	cout<<"Nodes per Element: "<<elem_nodes<<"\n";


	


	glutMainLoop();

	cudaDeviceReset();


	return 1;
    //exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);

}

