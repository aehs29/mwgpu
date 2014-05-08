#include <stdlib.h>
void matrixMult(float *A, float *B, float *C, int RowsC, int ColsC, int sharedDimension){
int i,j,k;
	//For Rows in A
	for(i=0;i<RowsC; i++){
		//For Columns in B
		for(j=0;j<ColsC;j++){
			int c=0;
			//Shared dimension
			for(k=0;k<sharedDimension;k++){
				//			k=k;
				//printf("Mult: %.2f by %.2f\n",h_A[i*(int)dimsA.x+k],h_B[k*dimsB.x+j]);
				c+=A[i*sharedDimension+k]*B[k*ColsC+j];
			}
			C[i*ColsC+j]=c;
		}
	}
}

void matrixByVec(float *mat, float *vec, float *res, int RowsMat, int ColsMat){
	int i,j;
	for (i=0;i<RowsMat;i++){
		float c=0;
		for(j=0;j<ColsMat;j++)
			c+=mat[i*ColsMat+j]*vec[j];
		res[i]=c;
	}
}

void matrixTranspose(float *A, float *B, int rows, int cols){
	for(int i=0;i<cols;i++)
		for(int j=0;j<rows;j++)
			// A[i,j]=B[j,i]
			A[i*rows+j]=B[j*cols+i];
}

void vectorAdd(float *A, float *B, float *C, int numElements){
	for(int i=0;i<numElements;i++)
		C[i]=A[i]+B[i];
}

void vectorbyScalar(float *A, const float scalar, int numElements){
	for(int i=0;i<numElements;i++)
		A[i]=A[i]*scalar;
}

void insertZeros(float *vec, int *positions , unsigned int fixed_nodes_count, unsigned int numElements){
	int i,j,k;
	numElements=numElements*3;
	for (i=0;i<fixed_nodes_count;i++){
		for (k=0;k<3;k++){
			for (j=numElements-1;j>positions[i]*3;j--){
				vec[j]=vec[j-1];
			}
			//for (k=0;k<3;k++)
			vec[(positions[i]*3)+k]=0;
		}
	}

}


void displacement_serial(float *q, float *qo, float *qd, float *qdo, float *F, float *Fo, float *R, float *Ro, float *alpha, float *alphaI, float *beta, float *gama, float *Phi, float *u, float h, unsigned int eigencount, unsigned int nodes_not_fixed, unsigned int node_dimensions, unsigned int nodes_original, int *fixed_nodes, float *nodes, float *nodes_orig){

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
	unsigned int size_nodes_orig = nodes_original*node_dimensions;
	unsigned int size_nodes = nodes_not_fixed * node_dimensions;
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

    // First part
    // eigenC x 1
	// u1=(alpha-Identity)*qo
	matrixByVec(alphaI,qo,u1,eigencount,eigencount);

	// Second part
	// eigenC x 1
	// u2=beta*qdo
	matrixByVec(beta,qdo,u2,eigencount,eigencount);

	// Third part
	// R is identity so u3=Phi
	std::copy(Phi,Phi+(size_nodes*size_eigen),u3);
	// Should create another copy after to get transpose

	// Third part .2
	// u3=transpose(u3)
	matrixTranspose(u3, Phi, size_nodes, size_eigen);

	// Third part .3
	// u4=u3*Fo
	matrixByVec(u3, Fo, u4, size_eigen, size_nodes);

	// Fourth part
	// u5=gama*u4
	// ERROR
	matrixByVec(gama, u4, u5, size_eigen, size_eigen);

	// Fifth part
	// u2=u2+u5
	vectorAdd(u2, u5, u2, size_eigen);

	// Actually get qd
	// qd=(u1+u2)/h
	vectorAdd(u1, u2, qd, size_eigen);
	vectorbyScalar(qd, 1/h, size_eigen);

	// Get q

	// First part
	// u1=alpha*qo
	matrixByVec(alpha,qo,u1,eigencount,eigencount);

	// Second part
	// q=u1+u2
	vectorAdd(u1, u2, q, size_eigen);

	// Calculate u
	// u=Phi*q
	matrixByVec(Phi,q,u,size_nodes,eigencount);

	// Add Zeros
	insertZeros(u, fixed_nodes , nodes_original-nodes_not_fixed, nodes_original);
	// Add u
	vectorAdd(u, nodes_orig, nodes, size_nodes_orig);
	// Free Memory
	free(u1);
	free(u2);
	free(u3);
	free(u4);
	free(u5);

}
