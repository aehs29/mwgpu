#include <stdlib.h>
void matrixMult(){

}

void matrixByVec(float *mat, float *vec, float *res, int RowsMat, int ColsMat){
	float c;
	for (int i=0;i<RowsMat;i++)
		for(int j=0;j<ColsMat;j++)
			c+=mat[i*ColsMat+j]*vec[j];
	res[i]=c;

}

void matrixTranspose(float *A, float *B, int rows, int cols){
	for(int i=0;i<rows;i++)
		for(int j=0;j<cols;j++)
			// A[i,j]=B[j,i]
			A[i*cols+j]=B[j*rows+i];
}

void vectorAdd(float *A, float *B, float *C, int numElements){
	for(int i=0;i<numElements;i++)
		C[i]=A[i]+B[i];
}

void vectorbyScalar(float *A, const float scalar, int numElements){
	for(int i=0;i<numElements;i++)
		A[i]=A[i]*scalar;
}

void displacement_serial(float *q, float *qo, float *qd, float *qdo, float *F, float *Fo, float *R, float *Ro, float *alpha, float *alphaI, float *beta, float *gama, float *Phi, float *u, float h, unsigned int eigencount, unsigned int node_count, unsigned int node_dimensions){

/*

alpha,beta,gama: eigencount x eigencount

q,qd: eigencount x 1

F, u: (node_count x node_dimensions) x 1

R: (node_count x node_dimensions) x (node_count x node_dimensions)

Phi:  (node_count x node_dimensions) x eigencount

*/
	// u1, u2: eigenC x 1 - same as q

	// u3: (node_count x node_dimensions) x eigencount - same as Phi

	// u4: eigenC x 1 - same as q

	// u5: eigenC x 1 - same as q


	// Mem Sizes
	unsigned int size_nodes = node_count* node_dimensions;
	unsigned int size_eigen = eigencount;
	unsigned int mem_size_q = sizeof(float) * size_eigen;
	unsigned int mem_size_Phi = sizeof(float) * (size_nodes*size_eigen);

	// Declare u's
	float *u1, *u2, *u3, *u4, *u5;

	// Allocate u's
	u1 = new float[size_eigen];
	u2 = new float[size_eigen];
	u3 = new float[size_nodes*size_eigen];
	u4 = new float[size_eigen];
	u5 = new float[size_eigen];




	// Get qd


	// Get q


	// Free Memory
	free(u1);
	free(u2);
	free(u3);
	free(u4);
	free(u5);

}
