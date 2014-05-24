// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cuPrintf.cu"


// helper CUDA functions
#include <helper_cuda.h>
#include <helper_functions.h>

__device__ float *d_original_nodes;


__device__ float *d_Phi, *d_Psy;

__device__ float *d_alphaI, *d_alpha, *d_beta, *d_gamma;



__global__ void kInsertZeros(float *Input, float *Output, unsigned int position, unsigned int number, unsigned int numElements) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i<numElements){
		// Do it 3 times
		// ibetween 3 & 5
		if(i>position-1 && i<=position+number-1)
			Output[i]=0;
		// i>=6
		else if(i>=position+number)
			Output[i]=Input[i-number];
		// i<3
		else if(i<position)
			Output[i]=Input[i];
	}
}

__global__ void kComputeR(float *d_w, float *d_uc, float *d_u , unsigned int numElements) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
   
	// Every thread does it for x y and z
	if(i>numElements)
		return;

	const int size=3;
	int rI=i*3;					// Real Index on w
	float norm,b,c;
	int j,k,l;
	float skew_w[size*size];
	float skew_w2[size*size];
	float R[size*size];

	// Check if its better to declare identity outside
	int Identity[size*size];
	for (j=0;j<size*size;j++){
		if(j==0 || j==4 || j==8)
			Identity[j]=1;
		else
			Identity[j]=0;
	}
	
	float sum=0;
	// Compute Norm
	for(j=0;j<size;j++){
		sum+=d_w[rI+j]*d_w[rI+j];	
		// if(i<3)
		// 	cuPrintf ("d_w[%d]:%f",rI+j, d_w[rI+j]);	
	}
	if(sum<0.00001)
		norm=1;
	else
		norm=sqrtf(sum);

	// if(i<3)
	// 	cuPrintf ("rI: %d, Norm: %f Sum:%.10f\n", rI,norm,sum);

	// Compute sine and cosine stuff
	b=cosf(norm);
	c=sinf(norm);

	b=(1-b)/norm;
	c=1-c/norm;

	// if(i<4){
	// 	cuPrintf("norm: %e, cos():%e, sin():%e\n",norm,b,c);
	// }
	

	// Check indices for skew matrix
    // Get Skew matrix				  
	for(j=0;j<size*size;j++){
		switch(j){
		case 0:
			skew_w[j]=0;
			break;
		case 1:
			skew_w[j]=-d_w[rI+2]/norm;
			break;
		case 2:
			skew_w[j]=d_w[rI+1]/norm;
			break;
		case 3:
			skew_w[j]=d_w[rI+2]/norm;
			break;
		case 4:
			skew_w[j]=0;
			break;
		case 5:
			skew_w[j]=-d_w[rI+0]/norm;
			break;
		case 6:
			skew_w[j]=-d_w[rI+1]/norm;
			break;
		case 7:
			skew_w[j]=d_w[rI+0]/norm;
			break;
		case 8:
			skew_w[j]=0;
			break;
		}
	}


	// if(i<4){
	// 	cuPrintf("norm: %e, cos():%e, sin():%e\n",norm,b,c);
	// }
	
	// Get skew matrix ^2
	for (j=0;j<size;j++){
		for (k=0;k<size;k++){
			sum=0;
			for(l=0;l<size;l++){
				sum+=skew_w[j*size+l]*skew_w[l*size+k];
			}
			skew_w2[j*size+k]=sum;
		}
	}	


	// Get R Matrix
	for(j=0;j<size*size;j++){
		R[j]=Identity[j]+skew_w[j]*b+skew_w2[j]*c;
	}


	// Multiply R by uc and modify u
	for (j=0;j<size;j++){
		sum=0;
		for(k=0;k<size;k++)
			sum+=R[j*size+k]*d_uc[rI+k];
		d_u[rI+j]=sum;
	}
	// if(i<3){
	// 	cuPrintf("u[0]: %e,u[0]: %e,u[0]: %e,\n",d_u[0],d_u[1],d_u[2]);
	// }

}


// Based on CUDA sdk example
template <int BLOCK_SIZE> __global__ void kMatrixMult(float *A, float *B, float *C,int ARows, int ACols, int BRows, int BCols)
{
	// Do it in blocks, accumulate in C
	int CRows=ARows;
	int CCols=BCols;
	
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	float CValue = 0;

    int Row = blockIdx.y*BLOCK_SIZE + ty;
    int Col = blockIdx.x*BLOCK_SIZE + tx;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
		if (k*BLOCK_SIZE + tx < ACols && Row < ARows)   
			As[ty][tx] = A[Row*ACols + k*BLOCK_SIZE + tx];
		else                                              
			As[ty][tx] = 0.0;
		
		if (k*BLOCK_SIZE + ty < BRows && Col < BCols)  
			Bs[ty][tx] = B[(k*BLOCK_SIZE + ty)*BCols + Col];
		else
			Bs[ty][tx] = 0.0;
		
		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) CValue += As[ty][n] * Bs[n][tx];

		__syncthreads();
    }

    if (Row < CRows && Col < CCols) 
		C[((blockIdx.y * blockDim.y + ty)*CCols)+(blockIdx.x*blockDim.x)+tx]=CValue;
}

__global__ void kVectorAdd(float *A, float *B, float *C, unsigned int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void kVectorScalar(float *A, const float scalar , int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        A[i] = A[i] * scalar;
    }
}

__global__ void kMatrixTranspose(float *A, float *B, int rows, int cols)
{
	int numElements=rows*cols;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	// j = current row
	int j = i/cols;
	// k =  current col
	int k = i-(j*cols);

    if (i < numElements)
    {
	    A[k*rows+j]=B[j*cols+k];
    }
}


__global__ void kMatVector(float *mat, float *vec, float *res, unsigned int RowsMat, unsigned int ColsMat) {
	float c = 0;
	//i is element in C to be computed
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i > RowsMat) return;

	for(int j=0;j<ColsMat;j++){
		c += mat[i*ColsMat+j]*vec[j];
	}
	res[i]=c;
}

__global__ void kModBuffer(float *buffer, float *d_nodes,  float *u, unsigned int totalThreads)
{
    const unsigned int TID = threadIdx.x;
    const unsigned int BID = blockIdx.x;
	const unsigned int BDIM = blockDim.x;

	unsigned int index=BDIM*BID+TID;
	if(index>totalThreads)
		return;
	// Add displacement to original nodes
	buffer[index]=d_nodes[index]+u[index];
	
}


extern "C" void map_Texture(void *cuda_dat, size_t siz,cudaGraphicsResource *resource)
{
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void **)(&cuda_dat), &size, resource);
}



extern "C" void allocate_GPUmem(float *nodes, float *h_alphaI, float *h_alpha, float *h_beta, float *h_gamma, float *h_eigenVecs, float *h_Psy, int node_count, int node_dimensions, int fixed_nodes_count,  int eigencount)
{
	unsigned int size_nodes = node_count * node_dimensions;
	unsigned int size_fixed = fixed_nodes_count * node_dimensions;


	unsigned int mem_size_nodes = sizeof(float) * size_nodes;

	unsigned int mem_size_coef = sizeof(float) * eigencount * eigencount;

	unsigned int mem_size_Phi = sizeof(float) * ((size_nodes-size_fixed)*eigencount);



	// Error code to check return values for CUDA calls
    cudaError_t error;


 

	// Allocate
	error = cudaMalloc((void **) &d_original_nodes, mem_size_nodes);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMalloc((void **) &d_alphaI, mem_size_coef);
	if (error != cudaSuccess){
        printf("cudaMalloc dd_alphaI returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMalloc((void **) &d_alpha, mem_size_coef);
	if (error != cudaSuccess){
        printf("cudaMalloc d_alpha returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMalloc((void **) &d_beta, mem_size_coef);
	if (error != cudaSuccess){
        printf("cudaMalloc d_beta returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMalloc((void **) &d_gamma, mem_size_coef);
	if (error != cudaSuccess){
        printf("cudaMalloc d_gama returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMalloc((void **) &d_Phi, mem_size_Phi);
	if (error != cudaSuccess){
        printf("cudaMalloc d_Phi returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMalloc((void **) &d_Psy, mem_size_Phi);
	if (error != cudaSuccess){
        printf("cudaMalloc d_Psy returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	// Copy
	
	error = cudaMemcpy(d_original_nodes, nodes, mem_size_nodes, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_nodes,h_nodes) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(d_alphaI, h_alphaI, mem_size_coef, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_nodes,h_nodes) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(d_alpha, h_alpha, mem_size_coef, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_alpha,h_alpha) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMemcpy(d_beta, h_beta, mem_size_coef, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_beta,h_beta) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(d_gamma, h_gamma, mem_size_coef, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_gama,h_gama) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(d_Phi, h_eigenVecs, mem_size_Phi, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_Phi,h_eigenVecs) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(d_Psy, h_Psy, mem_size_Phi, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_Phi,h_eigenVecs) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
}

extern "C" bool free_GPUnodes(){
	// Pretty sure this isn'd needed anymore
	// cudaGetSymbolAddress((void **)&d_nodes, d_original_nodes);
	// cudaFree(d_nodes);
	cudaFree(d_original_nodes);
	cudaFree(d_alphaI);
	cudaFree(d_alpha);
	cudaFree(d_beta);
	cudaFree(d_gamma);
	cudaFree(d_Phi);
	cudaFree(d_Psy);


	return true;
}



extern "C" bool displacement (float *h_q, float *h_qo, float *h_qd, float *h_qdo, float *h_F, float *h_Fo, float *h_Ro, float *h_alpha, float * h_alphaI, float *h_beta, float *h_gama, float *h_eigenVecs, float h_h, float *h_u, unsigned int eigencount, unsigned int node_count, unsigned int node_dimensions, const int block_size, float * buffer, float *h_nodes, int *fixed_nodes, unsigned int fixed_nodes_count, float *d_nodes, float *h_Psy){

    cudaPrintfInit ();


/*

  alpha,beta,gama: eigencount x eigencount

  q,qd: eigencount x 1

  F, u: (node_count x node_dimensions) x 1

  R: (node_count x node_dimensions) x (node_count x node_dimensions)

  Phi:  (node_count x node_dimensions) x eigencount

  NewPhi:  (node_count*node_dimensions)-(FixedNodes_count*node_dimensions) x eigencount

  Psy: Same as NewPhi

  w: same as u

  newF & R: 1 x (node_count*node_dimensions)-(FixedNodes_count*node_dimensions)

*/
	// u1, u2: eigenC x 1 - same as q

	// u3: (node_count x node_dimensions) x eigencount - same as Phi

	// u4: eigenC x 1 - same as q

	// u5: eigenC x 1 - same as q
	

/*#########---------Allocate variables---------------#########*/

	// Get size for pointers
	unsigned int size_nodes = node_count * node_dimensions;
	unsigned int size_eigen = eigencount;
	unsigned int size_fixed = fixed_nodes_count * node_dimensions;
	unsigned int size_nodestomodify = size_nodes-size_fixed;
	unsigned int mem_size_q = sizeof(float) * size_eigen;
	unsigned int mem_size_coef = sizeof(float) * (size_eigen*size_eigen);
	unsigned int mem_size_nodes = sizeof(float) * size_nodes;
	unsigned int mem_size_F = sizeof(float) * (size_nodes-size_fixed);
	unsigned int mem_size_R = sizeof(float) * (size_nodes*size_nodes);
	unsigned int mem_size_Phi = sizeof(float) * ((size_nodes-size_fixed)*size_eigen);

	// Number of zeros to insert in u array
	unsigned int numZeros = fixed_nodes_count *3;

    // Error code to check return values for CUDA calls
    cudaError_t error;

	const unsigned int maxThreadsBlock=512;
	const unsigned int warpSize=32;
	const unsigned int warpsBlock = maxThreadsBlock/warpSize;
	const unsigned int totalThreads = node_count*node_dimensions;
	const unsigned int warps = totalThreads/warpSize;
	const unsigned int mod = totalThreads%maxThreadsBlock;
	unsigned int blocks = warps/warpsBlock;
	unsigned int threadsDone = maxThreadsBlock*blocks;

	unsigned int threadsX;

	if (totalThreads>512)		// Depends on card
		threadsX=maxThreadsBlock;
	else{
		threadsX=totalThreads;
		blocks=1;
	}


	// Declare used variables on device
	float *d_qo, *d_qdo, *d_Ro, *d_Fo, *d_q, *d_qd, *d_w, *d_wc;

	// float *d_alphaI, *d_qo, *d_beta, *d_qdo, *d_Ro, *d_Phi, *d_Fo, *d_gama, *d_alpha, *d_q, *d_qd, *d_w, *d_wc, *d_Psy;

	// Declare u's on device
	float *d_u1, *d_u2, *d_u3, *d_u3c, *d_u4, *d_u5, *d_u, *d_uc;


/*#########---------Device Memory Allocation---------------#########*/ 

    error = cudaMalloc((void **) &d_u1, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u1 returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_u2, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u2 returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_u3, mem_size_Phi);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u3 returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMalloc((void **) &d_u3c, mem_size_Phi);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u3c returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_u4, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u4 returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_u5, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u5 returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_qo, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_qo returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_qdo, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_qdo returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    // error = cudaMalloc((void **) &d_Ro, mem_size_R);
	// if (error != cudaSuccess){
    //     printf("cudaMalloc d_Ro returned error code %d, line(%d)\n", error, __LINE__);
    //     exit(EXIT_FAILURE);
    // }

   

    error = cudaMalloc((void **) &d_Fo, mem_size_F);
	if (error != cudaSuccess){
        printf("cudaMalloc d_Fo returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_u, mem_size_nodes);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_qd, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_qd returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
	}

    error = cudaMalloc((void **) &d_q, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_q returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
	}
	error = cudaMalloc((void **) &d_uc, mem_size_nodes);
	if (error != cudaSuccess){
        printf("cudaMalloc d_uc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	
	error = cudaMalloc((void **) &d_w, mem_size_nodes);
	if (error != cudaSuccess){
        printf("cudaMalloc d_w returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMalloc((void **) &d_wc, mem_size_nodes);
	if (error != cudaSuccess){
        printf("cudaMalloc d_wc returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

/*#########---------Device Memory Allocation---------------#########*/ 


/*#########---------Host-to-Device Memory copy---------------#########*/ 

    error = cudaMemcpy(d_qo, h_qo, mem_size_q, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_qo,h_qo) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
   
    error = cudaMemcpy(d_qdo, h_qdo, mem_size_q, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_qdo,h_qdo) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }


    // error = cudaMemcpy(d_Ro, h_Ro, mem_size_R, cudaMemcpyHostToDevice);

    // if (error != cudaSuccess)
    // {
    //     printf("cudaMemcpy (d_Ro,h_Ro) returned error code %d, line(%d)\n", error, __LINE__);
    //     exit(EXIT_FAILURE);
    // }
   
    error = cudaMemcpy(d_Fo, h_Fo, mem_size_F, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_Fo,h_Fo) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    

   
/*#########---------Host-to-Device Memory copy---------------#########*/ 



/*#########---------Ends variables allocation---------------#########*/



/*#########---------Get qd---------------#########*/

/*###----------------First Part--------------------####*/ 
	// First part
	// eigenC x 1
	// u1=(alpha-Identity)*qo       Can be done with VectorAdd Kernel, say whaat?

	// error = cudaMemcpy(h_u, d_u, mem_size_nodes, cudaMemcpyDeviceToHost);

	//Launch one extra block to make it multiple of 32
	unsigned int threadsPB= 512;
	unsigned int BPG = (eigencount + threadsPB - 1) / threadsPB;
	dim3 threadsPerBlock(threadsPB);
	dim3 blocksPerGrid(BPG);

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_alphaI, d_qo, d_u1, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u1) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


	// remove this

	// error = cudaMemcpy(h_q, d_u1, mem_size_q, cudaMemcpyDeviceToHost);

    // if (error != cudaSuccess)
    // {
    //     printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
    //     exit(EXIT_FAILURE);
    // }

//--------------------Second Part-------------------	

	// Second part
	// eigenC x 1
	// u2=beta*qdo

	// Same kernel parameters as for u1

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_beta, d_qdo, d_u2, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// remove this
	// error = cudaMemcpy(h_q, d_u2, mem_size_q, cudaMemcpyDeviceToHost);

    // if (error != cudaSuccess)
    // {
    //     printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
    //     exit(EXIT_FAILURE);
    // }

//-------------------------Third Part .1 --------------------------	



	// Very OLD code - Need it later

	// Third part should be in a different variable to use it again on q

	// Third part .1
	// (node_count x node_dimensions) x eigencount
    // u3 = (Ro*Phi)

// 	unsigned int ColsB, RowsA;
// 	ColsB=eigencount;
// 	RowsA=size_nodes;

// 	// Different variables than MatByVec kernel
//     dim3 threads(block_size, block_size);
//     dim3 grid((ColsB+threads.x-1) / threads.x, (RowsA+threads.y-1) / threads.y);
// // Exec kernel
// 	kMatrixMult<16><<< grid, threads >>>(d_Ro, d_Phi, d_u3, size_nodes,size_nodes, size_nodes, size_eigen);
// 	    error = cudaGetLastError();
//     if (error != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to launch matrixMult kernel (u3) (error code %s)!\n", cudaGetErrorString(error));
//         exit(EXIT_FAILURE);
//     }

	// FOR NOW U3=Phi
	//d_u3=d_Phi;


//-------------------------Third Part .2-----------------------------

	// Third part .2
	// eigencount x (node_count x node_dimensions)
    // u3 = Transpose(u3)

	// Make a copy of u3 to transpose it
	error = cudaMemcpy(d_u3c, d_Phi, mem_size_Phi, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_u3c,d_u3) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	// remove this
	// error = cudaMemcpy(h_Ro, d_u3c, mem_size_Phi, cudaMemcpyDeviceToHost);

    // if (error != cudaSuccess)
    // {
    //     printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
    //     exit(EXIT_FAILURE);
    // }

	// New kernel parameters
	unsigned int numElements = (size_nodes-size_fixed)*size_eigen;
	blocksPerGrid=(numElements + threadsPB - 1) / threadsPB;

	// Arguments: Result, Original, # Elements in Matrix
    kMatrixTranspose<<< blocksPerGrid, threadsPerBlock>>>(d_u3,d_u3c, size_nodes-size_fixed, size_eigen);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch matrixTranspose kernel (u3) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


	// remove this
	// error = cudaMemcpy(h_Ro, d_u3, mem_size_Phi, cudaMemcpyDeviceToHost);

    // if (error != cudaSuccess)
    // {
    //     printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
    //     exit(EXIT_FAILURE);
    // }

//------------------------Third part .3-----------------------------
	// Third part .3
	// eigencount x 1
    // u4 = u3*Fo
	blocksPerGrid=(eigencount + threadsPB - 1) / threadsPB;

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_u3, d_Fo, d_u4, eigencount, size_nodes-size_fixed);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u4) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// Remove this
	// error = cudaMemcpy(h_q, d_u4, mem_size_q, cudaMemcpyDeviceToHost);

    // if (error != cudaSuccess)
    // {
    //     printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
    //     exit(EXIT_FAILURE);
    // }
//-----------------------Part 4-----------------------
	// eigencount x 1
    // u5 = gama*u4

	// Same kernel params as u4

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_gamma, d_u4, d_u5, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u5) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// remove this
	// error = cudaMemcpy(h_q, d_u5, mem_size_q, cudaMemcpyDeviceToHost);

    // if (error != cudaSuccess)
    // {
    //     printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
    //     exit(EXIT_FAILURE);
    // }

//--------------------------------------------
	// Fourth part sum all and divive by h
	// u2=u2+u5
	// vector + vector kernel u2=u2+u5

	// Same kernel parameters

	// Arguments: Vector1, Vector2, VectorResult,  # of Elements
	kVectorAdd<<< blocksPerGrid, threadsPerBlock>>>(d_u2, d_u5, d_u2, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u2.2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// remove this
	// error = cudaMemcpy(h_q, d_u2, mem_size_q, cudaMemcpyDeviceToHost);

	// qd=(u1+u2)/h

	// vector + vector u1=u1+u2
	// Same kernel parameters

	// Arguments: Vector1, Vector2, VectorResult,  # of Elements
	kVectorAdd<<< blocksPerGrid, threadsPerBlock>>>(d_u1, d_u2, d_qd, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch VectorAdd kernel (u1.2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// vector by scalar kernel
	// qd=u1*1/h

	// Arguments: vector, scalar, # elements
    kVectorScalar<<< blocksPerGrid, threadsPerBlock>>>(d_qd, 1/h_h,  eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


	// remove this
	// error = cudaMemcpy(h_q, d_qd, mem_size_q, cudaMemcpyDeviceToHost);

/*#########---------Ends Get qd---------------#########*/


/*#########---------Get q---------------#########*/

	// First part
	// eigenC x 1
	// u1=alpha*qo       Can be done with VectorAdd Kernel, say whaat?

	// Same parameters as other kernels

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_alpha, d_qo, d_u1, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u1.q) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// remove this
	// error = cudaMemcpy(h_q, d_u1, mem_size_q, cudaMemcpyDeviceToHost);

	// Second part
	// eigenC x 1
	// q=u1+u2
    // Same kernel parameters

	// Arguments: Vector1, Vector2, VectorResult,  # of Elements
	kVectorAdd<<< blocksPerGrid, threadsPerBlock>>>(d_u1, d_u2, d_q, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch VectorAdd kernel (u1.2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	// remove this
	// error = cudaMemcpy(h_q, d_q, mem_size_q, cudaMemcpyDeviceToHost);

	
/*#########---------Ends Get q---------------#########*/

/*#########---------Calculate u---------------#########*/

	//   (node_count x node_dimensions) x 1
	// u=Phi *q

	blocksPerGrid=(size_nodes-size_fixed + threadsPB - 1) / threadsPB;

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_Phi, d_q, d_u, size_nodes-size_fixed, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// remove this
	// error = cudaMemcpy(h_u, d_u, mem_size_nodes, cudaMemcpyDeviceToHost);

/*#########---------Ends Calculate u---------------#########*/

/*#########---------Calculate w---------------#########*/

	//   (node_count x node_dimensions) x 1
	// u=Phi *q

	blocksPerGrid=(size_nodes-size_fixed + threadsPB - 1) / threadsPB;

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_Psy, d_q, d_w, size_nodes-size_fixed, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// remove this
	// error = cudaMemcpy(h_u, d_u, mem_size_nodes, cudaMemcpyDeviceToHost);

/*#########---------Ends Calculate w---------------#########*/


/*#########---------Compute R for every node and modify u---------------#########*/

	error = cudaMemcpy(d_uc, d_u, size_nodestomodify*sizeof(float), cudaMemcpyDeviceToDevice);

	// Each thread computes skew matrix for every node (x,y,z - 3 elements)
	blocksPerGrid =(node_count-fixed_nodes_count + threadsPB - 1) / threadsPB;
    // error = cudaMemcpy(h_u, d_w, mem_size_nodes, cudaMemcpyDeviceToHost);
	
	// for (int g=0;g<9;g++)
	// 	printf("w[%d]: %f\n",g,h_u[g]);

	
	kComputeR<<< blocksPerGrid, threadsPerBlock>>>(d_w, d_uc, d_u,node_count-fixed_nodes_count); 

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kComputeR kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// error = cudaMemcpy(h_u, d_u, mem_size_nodes, cudaMemcpyDeviceToHost);


/*#########---------Ends Compute R---------------#########*/



/*#########---------Insert Zeros---------------#########*/

	blocksPerGrid =(size_nodes + threadsPB - 1) / threadsPB;
	unsigned int position;
	int i;
   
	// Insert 0's on desired locations
	for(i=0;i<fixed_nodes_count;i++){
		position = fixed_nodes[i]*3;
		// Arguments: Input Array, Output per kernel, poisiton of first 0, # of consecutive 0's
		// Reuse arrays to save memory
		if(i%2==1){
			kInsertZeros<<< blocksPerGrid, threadsPerBlock>>>(d_uc, d_u, position, 3, size_nodestomodify+(i+1)*3);
			// kInsertZeros<<< blocksPerGrid, threadsPerBlock>>>(d_wc, d_w, position, 3, size_nodestomodify+(i+1)*3);

			// remove this
			// error = cudaMemcpy(h_u, d_u, mem_size_nodes, cudaMemcpyDeviceToHost);
		}
		else{
			kInsertZeros<<< blocksPerGrid, threadsPerBlock>>>(d_u, d_uc, position, 3, size_nodestomodify+(i+1)*3);
			// kInsertZeros<<< blocksPerGrid, threadsPerBlock>>>(d_w, d_wc, position, 3, size_nodestomodify+(i+1)*3);

			// remove this
			// error = cudaMemcpy(h_u, d_uc, mem_size_nodes, cudaMemcpyDeviceToHost);
		}

		error = cudaGetLastError();

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch InsertZeros kernel (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}


	// Copy correct array - Now with R calculation included both original and copy must be the same
	if(i%2==0){
		error = cudaMemcpy(d_u, d_uc, size_nodes, cudaMemcpyDeviceToDevice);
		// error = cudaMemcpy(d_w, d_wc, size_nodes, cudaMemcpyDeviceToDevice);
	}
	else{
		error = cudaMemcpy(d_uc, d_u, size_nodes, cudaMemcpyDeviceToDevice);
		// error = cudaMemcpy(d_wc, d_w, size_nodes, cudaMemcpyDeviceToDevice);
	}

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output from device to host (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// remove this
	// error = cudaMemcpy(h_u, d_u, mem_size_nodes, cudaMemcpyDeviceToHost);
	// error = cudaMemcpy(h_u, d_uc, mem_size_nodes, cudaMemcpyDeviceToHost);
	// if (error != cudaSuccess)
	// {
	// 	fprintf(stderr, "Failed to copy output from device to host (error code %s)!\n", cudaGetErrorString(error));
	// 	exit(EXIT_FAILURE);
	// }

/*#########---------Ends Insert Zeros---------------#########*/




/*#########---------Render on OpenGL---------------#########*/


	// TODO change variable names
    dim3 grid(blocks+1, 1, 1);
    dim3 threads(threadsX, 1, 1);
	kModBuffer<<< grid, threads >>>(buffer, d_original_nodes, d_u, size_nodes);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch ModBuffer kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

/*#########---------Render on OpenGL---------------#########*/


/*#########---------Free Memory---------------#########*/
	// Copy result from device to host

    // This isnt needed?
    error = cudaMemcpy(h_u, d_u, mem_size_nodes, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_u,d_u) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Should copy to hqo instead
    error = cudaMemcpy(h_q, d_q, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Same, should copy to old array
    error = cudaMemcpy(h_qd, d_qd, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_qd,d_qd) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	cudaFree(d_u1);
	cudaFree(d_u2);
	cudaFree(d_u3);
	cudaFree(d_u3c);
	cudaFree(d_u4);
	cudaFree(d_u5);
	// cudaFree(d_alphaI);
	cudaFree(d_qo);
	cudaFree(d_qdo);
	cudaFree(d_Fo);
	cudaFree(d_u);
	cudaFree(d_uc);
	cudaFree(d_q);
	cudaFree(d_qd);
	cudaFree(d_w);


    // CudaPrintf Stuff
	cudaPrintfDisplay (stdout, true);
    cudaPrintfEnd ();

	bool success=true;
	if(error != cudaSuccess)
		success=false;
	return success;
		

/*#########---------Ends Free Memory---------------#########*/
}
