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

// Original Nodes
__device__ float *d_original_nodes;

// Matrices
__device__ float *d_Phi, *d_Psi;

// Coefficient Matrices
__device__ float *d_alphaI, *d_alpha, *d_beta, *d_gamma;

// u's
__device__ 	float *d_u1, *d_u2, *d_PhiT, *d_u4, *d_u5, *d_u, *d_uc;

// Calculated vars
float *d_qo, *d_qdo, *d_Fo, *d_w, *d_wc;

// Time measurement
cudaEvent_t start;
cudaEvent_t stop;

__global__ void kInsertZeros(float *Input, float *Output, unsigned int position, unsigned int number, unsigned int numElements) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i<numElements){
		// Do it 3 times
		// i between 3 & 5
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
	}
	if(sum<0.00001)
		norm=1;
	else
		norm=sqrtf(sum);

	// Compute sine and cosine stuff
	b=cosf(norm);
	c=sinf(norm);

	b=(1-b)/norm;
	c=1-c/norm;

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


__global__ void kMatVector(float *mat, float *vec, float *res, unsigned int RowsMat, const unsigned int ColsMat) {
	float c = 0;
	//i is element in C to be computed
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	
	if(i > RowsMat) return;

	// extern __shared__ float vecsh[];
	//  for(int j=0;j<ColsMat;j++){
	//  	vecsh[j]=vec[j];
	//  }

	//  __syncthreads();

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



extern "C" void allocate_GPUmem(float *nodes, float *h_alphaI, float *h_alpha, float *h_beta, float *h_gamma, float *h_eigenVecs, float *h_Psi,int node_count, int node_dimensions, int fixed_nodes_count,  int eigencount)
{
	unsigned int size_nodes = node_count * node_dimensions;
	unsigned int size_fixed = fixed_nodes_count * node_dimensions;
	unsigned int mem_size_nodes = sizeof(float) * size_nodes;
	unsigned int mem_size_coef = sizeof(float) * eigencount * eigencount;
	unsigned int mem_size_Phi = sizeof(float) * ((size_nodes-size_fixed)*eigencount);
	unsigned int mem_size_q = sizeof(float) * eigencount;
	unsigned int mem_size_F = sizeof(float) * (size_nodes-size_fixed);

	// Error code to check return values for CUDA calls
    cudaError_t error;
 
/*#########---------Device Memory Allocation---------------#########*/ 
	
	// error = cudaHostRegister(h_F, mem_size_F, cudaHostRegisterMapped);
	// if (error != cudaSuccess){
	// 	printf("cudaHostRegister h_F returned error code %d, line(%d)\n", error, __LINE__);
    //     exit(EXIT_FAILURE);
    // }
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

	error = cudaMalloc((void **) &d_Psi, mem_size_Phi);
	if (error != cudaSuccess){
        printf("cudaMalloc d_Psi returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

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
	error = cudaMalloc((void **) &d_PhiT, mem_size_Phi);
	if (error != cudaSuccess){
        printf("cudaMalloc d_PhiT returned error code %d, line(%d)\n", error, __LINE__);
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

   
    error = cudaMalloc((void **) &d_u, mem_size_nodes);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u returned error code %d, line(%d)\n", error, __LINE__);
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
	error = cudaMalloc((void **) &d_Fo, mem_size_F);
	if (error != cudaSuccess){
        printf("cudaMalloc d_Fo returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
/*#########---------Device Memory Allocation---------------#########*/ 


/*#########---------Host-to-Device Memory copy---------------#########*/ 
	
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
	error = cudaMemcpy(d_Psi, h_Psi, mem_size_Phi, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_Phi,h_eigenVecs) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	cudaMemset(d_qo,0,mem_size_q);
	cudaMemset(d_qdo,0,mem_size_q);

    // Third part .2
	// eigencount x (node_count x node_dimensions)
    // u3 = Transpose(u3)

	unsigned int threadsPB= 512;

	// Make a copy of u3 to transpose it
	// error = cudaMemcpy(d_PhiT, d_Phi, mem_size_Phi, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_u3c,d_u3) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }


	// New kernel parameters
	unsigned int numElements = (size_nodes-size_fixed)*eigencount;

	dim3 blocksPerGrid((numElements + threadsPB - 1) / threadsPB);
	dim3 threadsPerBlock(threadsPB);

	// Arguments: Result, Original, # Elements in Matrix
    kMatrixTranspose<<< blocksPerGrid, threadsPerBlock>>>(d_PhiT,d_Phi, size_nodes-size_fixed, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch matrixTranspose kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


}

extern "C" bool free_GPUnodes(){
	// Free globals on GPU
	cudaFree(d_original_nodes);
	cudaFree(d_alphaI);
	cudaFree(d_alpha);
	cudaFree(d_beta);
	cudaFree(d_gamma);
	cudaFree(d_Phi);
	cudaFree(d_Psi);
	cudaFree(d_u1);
	cudaFree(d_u2);
	cudaFree(d_PhiT);
	cudaFree(d_u4);
	cudaFree(d_u5);
	cudaFree(d_qo);
	cudaFree(d_qdo);
	cudaFree(d_Fo);
	cudaFree(d_u);
	cudaFree(d_uc);
	cudaFree(d_w);
	return true;
}



extern "C" bool displacement (float *h_Fo, float h_h, unsigned int eigencount, unsigned int node_count, unsigned int node_dimensions, float *simtime, const int block_size, float *buffer, int *fixed_nodes, unsigned int fixed_nodes_count, unsigned int maxThreadsBlock, bool debug){

/*

  alpha,beta,gama: eigencount x eigencount

  q,qd: eigencount x 1

  F, u: (node_count x node_dimensions) x 1

  R: (node_count x node_dimensions) x (node_count x node_dimensions)

  Phi:  (node_count x node_dimensions) x eigencount

  NewPhi:  (node_count*node_dimensions)-(FixedNodes_count*node_dimensions) x eigencount

  Psi: Same as NewPhi

  w: same as u

  newF & R: 1 x (node_count*node_dimensions)-(FixedNodes_count*node_dimensions)

*/
	// u1, u2: eigenC x 1 - same as q

	// u3: (node_count x node_dimensions) x eigencount - same as Phi

	// u4: eigenC x 1 - same as q

	// u5: eigenC x 1 - same as q


	// Initialize Printf on CUDA
    cudaPrintfInit ();

	// Create and start timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	float tmpTime,totTime=0;

/*#########---------Allocate variables---------------#########*/

	// Get size for pointers
	unsigned int size_nodes = node_count * node_dimensions;
	// unsigned int size_eigen = eigencount;
	unsigned int size_fixed = fixed_nodes_count * node_dimensions;
	unsigned int size_freenodes = size_nodes-size_fixed;
	// unsigned int mem_size_q = sizeof(float) * size_eigen;
	unsigned int mem_size_F = sizeof(float) * (size_nodes-size_fixed);
	// unsigned int mem_size_Phi = sizeof(float) * ((size_nodes-size_fixed)*size_eigen);
	unsigned int mem_size_freenodes = size_freenodes * sizeof(float);

    // Error code to check return values for CUDA calls
    cudaError_t error;

	// Used for rendering
	// const unsigned int maxThreadsBlock=512;
	const unsigned int warpSize=32;
	const unsigned int warpsBlock = maxThreadsBlock/warpSize;
	const unsigned int totalThreads = node_count*node_dimensions;
	const unsigned int warps = totalThreads/warpSize;

	// # Threads to be used in kernels
	unsigned int threadsPB= maxThreadsBlock;

	// cudaStream_t s1,s2,s3;
	// cudaStreamCreate(&s1);
	// cudaStreamCreate(&s2);
	// cudaStreamCreate(&s3);

	int nstreams= 3;
   // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));

    for (int i = 0; i < nstreams; i++)
    {   
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }   


	// Time measure for debugging
	if(debug==true){

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"Initial declaration:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}

/*#########---------Host-to-Device Memory copy---------------#########*/ 
   
    // error = cudaMemcpyAsync(d_Fo, h_Fo, mem_size_F, cudaMemcpyHostToDevice);
    error = cudaMemcpy(d_Fo, h_Fo, mem_size_F, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_Fo,h_Fo) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	// cudaStreamDestroy(s1);
/*#########---------Host-to-Device Memory copy---------------#########*/ 

	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"Host2Device MemCpy:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}


/*#########---------Get qd---------------#########*/

/*###----------------First Part--------------------####*/ 
	// First part
	// eigenC x 1
	// u1=(alpha-Identity)*qo

	//Launch one extra block to make it multiple of 32
	unsigned int BPG = (eigencount + threadsPB - 1) / threadsPB;
	dim3 threadsPerBlock(threadsPB);
	dim3 blocksPerGrid(BPG);

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock,0,streams[0]>>>(d_alphaI, d_qo, d_u1, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u1) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"qd - Part 1:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}

//--------------------Second Part-------------------	

	// Second part
	// eigenC x 1
	// u2=beta*qdo

	// Same kernel parameters as for u1

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock,0,streams[1]>>>(d_beta, d_qdo, d_u2, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"qd - Part 2:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}
//-------------------------Third Part .1 --------------------------	
   
//------------------------Third part .3-----------------------------
	// Third part .3
	// eigencount x 1
    // u4 = u3*Fo
	blocksPerGrid=(eigencount + threadsPB - 1) / threadsPB;

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_PhiT, d_Fo, d_u4, eigencount, size_nodes-size_fixed);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u4) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"qd - Part 3 - Mult:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}
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

	// Arguments: Vector1, Vector2, VectorResult,  # of Elements
	kVectorAdd<<< blocksPerGrid, threadsPerBlock>>>(d_u1, d_u2, d_qdo, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch VectorAdd kernel (u1.2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// vector by scalar kernel
	// qd=u1*1/h

	// Arguments: vector, scalar, # elements
    kVectorScalar<<< blocksPerGrid, threadsPerBlock,0,streams[2]>>>(d_qdo, 1/h_h,  eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"qd - Part 4:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}

/*#########---------Ends Get qd---------------#########*/


/*#########---------Get q---------------#########*/

	// First part
	// eigenC x 1
	// u1=alpha*qo

	// Same parameters as other kernels

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock,0,streams[0]>>>(d_alpha, d_qo, d_u1, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u1.q) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"q - Part 1:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}
	// Second part
	// eigenC x 1
	// q=u1+u2
    // Same kernel parameters

	// Arguments: Vector1, Vector2, VectorResult,  # of Elements
	kVectorAdd<<< blocksPerGrid, threadsPerBlock>>>(d_u1, d_u2, d_qo, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch VectorAdd kernel (u1.2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"q - Part 2:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}
	
/*#########---------Ends Get q---------------#########*/

/*#########---------Calculate u---------------#########*/

	//   (node_count x node_dimensions) x 1
	// u=Phi *q

	blocksPerGrid=(size_nodes-size_fixed + threadsPB - 1) / threadsPB;

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock,0,streams[0]>>>(d_Phi, d_qo, d_u, size_nodes-size_fixed, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"u - Calculation:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}

/*#########---------Ends Calculate u---------------#########*/

/*#########---------Calculate w---------------#########*/

	//   (node_count x node_dimensions) x 1
	// u=Psi *q

	// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock,0,streams[1]>>>(d_Psi, d_qo, d_w, size_nodes-size_fixed, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"w - Calculation:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}

/*#########---------Ends Calculate w---------------#########*/


/*#########---------Compute R for every node and modify u---------------#########*/

	error = cudaMemcpy(d_uc, d_u, mem_size_freenodes, cudaMemcpyDeviceToDevice);

	// Each thread computes skew matrix for every node (x,y,z - 3 elements)
	blocksPerGrid =(node_count-fixed_nodes_count + threadsPB - 1) / threadsPB;
	
    // Arguments: w vector, copy of u, u, nodes to calculate R for
	kComputeR<<< blocksPerGrid, threadsPerBlock>>>(d_w, d_uc, d_u,node_count-fixed_nodes_count); 

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kComputeR kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"R - Calculation:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}

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
			kInsertZeros<<< blocksPerGrid, threadsPerBlock>>>(d_uc, d_u, position, 3, size_freenodes+(i+1)*3);
		}
		else{
			kInsertZeros<<< blocksPerGrid, threadsPerBlock>>>(d_u, d_uc, position, 3, size_freenodes+(i+1)*3);
		}

		error = cudaGetLastError();

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch InsertZeros kernel (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}


	// Copy correct array
	if(i%2==0){
		error = cudaMemcpy(d_u, d_uc, size_nodes, cudaMemcpyDeviceToDevice);
	}

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output from device to host (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"Insert Zeros:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}
/*#########---------Ends Insert Zeros---------------#########*/


/*#########---------Render on OpenGL---------------#########*/

	
	blocksPerGrid = 1+(warps/warpsBlock);

	kModBuffer<<< blocksPerGrid, threadsPerBlock >>>(buffer, d_original_nodes, d_u, size_nodes);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch ModBuffer kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Time measure for debugging
	if(debug==true){
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&tmpTime,start,stop);
		totTime+=tmpTime;
		std::cout<<"Modify Buffer:\t"<<tmpTime<<"\n";
		cudaEventRecord(start,0);
	}
/*#########---------Render on OpenGL---------------#########*/

	// Time measure for debugging
	if(debug==true){
		*simtime=totTime;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		std::cout<<"Total Simulation Time:\t"<<totTime<<"\n\n\n";
	}
	else{
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(simtime,start,stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
    // CudaPrintf Stuff
	cudaPrintfDisplay (stdout, true);
    cudaPrintfEnd ();

	bool success=true;
	if(error != cudaSuccess)
		success=false;
	return success;
		

/*#########---------Ends Free Memory---------------#########*/
}
