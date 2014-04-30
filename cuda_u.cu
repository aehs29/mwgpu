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



// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void kern(int i)
{

i=i+i;
}

template <int BLOCK_SIZE> __global__ void kMatrixMult(float *A, float *B, float *C,int ARows, int ACols, int BRows, int BCols)
{

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
		// A[j,k]=B[k,j]
		// A[k*rows+j]=B[j*rows+k]
		// if j=2, k =3
		// A[2,3]=12
		// B[3,2]=15
	    A[k*rows+j]=B[j*cols+k];
    }
}


__global__ void kMatVector(float *mat, float *vec, float *res, unsigned int RowsMat, unsigned int ColsMat) {
// Each thread computes one element of C
// by accumulating results into Cvalue
	float c = 0;
	//i is element in C to be computed
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i > RowsMat) return;

//    if (i < ColsA)
//    {
		for(int j=0;j<ColsMat;j++){
			c += mat[i*ColsMat+j]*vec[j];
		}
		res[i]=c;
//    }


}



//__global__ void kernel(float scale)
__global__ void kernel(float *g_nodes, float *u, float scale)
{
    // write data to global memory
    const unsigned int TID = threadIdx.x;
    const unsigned int BID = blockIdx.x;
	const unsigned int BDIM = blockDim.x;
//    float nodes = g_nodes[tid];

    // use integer arithmetic to process all four bytes with one thread
    // this serializes the execution, but is the simplest solutions to avoid
    // bank conflicts for this very low number of threads
    // in general it is more efficient to process each byte by a separate thread,
    // to avoid bank conflicts the access pattern should be
    // g_data[4 * wtid + wid], where wtid is the thread id within the half warp
    // and wid is the warp id
    // see also the programming guide for a more in depth discussion.
//    g_data[tid] = ((((data <<  0) >> 24) - 10) << 24)
//                  | ((((data <<  8) >> 24) - 10) << 16)
//                  | ((((data << 16) >> 24) - 10) <<  8)
//                  | ((((data << 24) >> 24) - 10) <<  0);

	// g_nodes[BDIM*BID+TID]=(g_nodes[BDIM*BID+TID])*scale;
	g_nodes[BDIM*BID+TID]=(g_nodes[BDIM*BID+TID]+u[BDIM*BID+TID])*scale;
//	scale*scale;
}
__global__ void kernelRemaining(float *g_nodes, float *u, float scale, unsigned int threadsDone)
{
    // write data to global memory
    const unsigned int TID = threadIdx.x;

	// g_nodes[threadsDone-1+TID]=(g_nodes[threadsDone-1+TID])*scale;
	g_nodes[threadsDone-1+TID]=(g_nodes[threadsDone-1+TID]+u[threadsDone-1+TID])*scale;
}

//float *cuda_data=NULL;
extern "C" void map_texture(void *cuda_dat, size_t siz,cudaGraphicsResource *resource)
{
size_t size;
cudaGraphicsResourceGetMappedPointer((void **)(&cuda_dat), &size, resource);
}

extern "C" bool
runTest(const int argc, const char **argv, float *buffer, float *u, int node_count, float scale)
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //findCudaDevice(argc, (const char **)argv);



	// CUDA
//cudaSetDevice(0);

//	cudaGLSetGLDevice(0);



const unsigned int maxThreadsBlock=512;
const unsigned int warpSize=32;
const unsigned int warpsBlock = maxThreadsBlock/warpSize;
const unsigned int totalThreads = node_count*3;
const unsigned int warps = totalThreads/warpSize;
const unsigned int mod = totalThreads%maxThreadsBlock;
unsigned int blocks = warps/warpsBlock;
unsigned int threadsDone = maxThreadsBlock*blocks;

unsigned int threadsX;

	if (totalThreads>512)
		threadsX=maxThreadsBlock;
	else{
		threadsX=totalThreads;
		blocks=1;
	}

    //const unsigned int num_threads = (node_count*3);
//	const unsigned int num_threads = 1;
//    assert(0 == (len % 4));
    const unsigned int mem_size = sizeof(float) * totalThreads;
//    const unsigned int mem_size_int2 = sizeof(int2) * len;

    // allocate device memory
    //float *d_nodes;
    //checkCudaErrors(cudaMalloc((void **) &d_nodes, mem_size));
    // copy host memory to device
    //checkCudaErrors(cudaMemcpy(d_nodes, nodes, mem_size,
    //                           cudaMemcpyHostToDevice));
    // allocate device memory for int2 version
//    int2 *d_data_int2;
//    checkCudaErrors(cudaMalloc((void **) &d_data_int2, mem_size_int2));
    // copy host memory to device
 //   checkCudaErrors(cudaMemcpy(d_data_int2, data_int2, mem_size_int2,
  //                             cudaMemcpyHostToDevice));

	//const int y=num_threads%512;

    cudaError_t error;

	unsigned int mem_size_nodes = sizeof(float) * node_count*3;

	float *d_u;

	error = cudaMalloc((void **) &d_u, mem_size_nodes);
	if (error != cudaSuccess){
        printf("cudaMalloc d_u returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(d_u, u, mem_size_nodes, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_u,u) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }



    // setup execution parameters
    dim3 grid(blocks, 1, 1);
    dim3 threads(threadsX, 1, 1);
    //dim3 threads2(len, 1, 1); // more threads needed fir separate int2 version
//	printf("Executing Kernel, Threads: %u, Scale:%f\n",num_threads,scale);
    // execute the kernel
//    kernel<<< grid, threads >>>(scale);

/*
printf("Blocks:%u\n",blocks);
printf("ThreadsOnK:%u\n",threadsX);
printf("Remaining:%u\n",mod);
printf("threadsDone:%u\n", threadsDone);
*/
    kernel<<< grid, threads >>>(buffer, d_u, scale);
if(totalThreads>512)
	kernelRemaining<<< 1,mod >>>(buffer, d_u, scale, threadsDone);
    //kernel2<<< grid, threads2 >>>(d_data_int2);

	
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // compute reference solutions
//    char *reference = (char *) malloc(mem_size);
//    computeGold(reference, data, len);
//    int2 *reference2 = (int2 *) malloc(mem_size_int2);
//    computeGold2(reference2, data_int2, len);

    // copy results from device to host
   // checkCudaErrors(cudaMemcpy(nodes, d_nodes, mem_size,
     //                          cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaMemcpy(data_int2, d_data_int2, mem_size_int2,
//                               cudaMemcpyDeviceToHost));

    // check result
    bool success = true;

//    for (unsigned int i = 0; i < len; i++)
//    {
//        if (reference[i] != data[i] ||
//            reference2[i].x != data_int2[i].x ||
//            reference2[i].y != data_int2[i].y)
//        {
//            success = false;
//        }
//    }

    // cleanup memory
//    checkCudaErrors(cudaFree(d_nodes));
//    checkCudaErrors(cudaFree(d_data_int2));
//    free(reference);
//    free(reference2);

    return success;
}

extern "C" bool runTest2(const int argc, const char **argv, int i)
{
	cudaSetDevice(0);
//	cudaGLSetGLDevice(0);
  dim3 grid(1, 1, 1);
    dim3 threads(2, 1, 1);

	  kern<<< grid, threads >>>(i);
    getLastCudaError("Kernel execution failed");

	  std::cout<<i;
	  return true;
}




extern "C" bool displacement (float *h_q, float *h_qo, float *h_qd, float *h_qdo, float *h_F, float *h_Fo, float *h_Ro, float *h_alpha, float * h_alphaI, float *h_beta, float *h_gama, float *h_eigenVecs, float h_h, float *h_u, unsigned int eigencount, unsigned int node_count, unsigned int node_dimensions, const int block_size){


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


/*#########---------Allocate variables---------------#########*/

	// Get size for pointers
	unsigned int size_nodes = node_count* node_dimensions;
	unsigned int size_eigen = eigencount;
	unsigned int mem_size_q = sizeof(float) * size_eigen;
	unsigned int mem_size_coef = sizeof(float) * (size_eigen*size_eigen);
	unsigned int mem_size_nodes = sizeof(float) * size_nodes;
	unsigned int mem_size_R = sizeof(float) * (size_nodes*size_nodes);
	unsigned int mem_size_Phi = sizeof(float) * (size_nodes*size_eigen);


	// Declare used variables on device
	float *d_alphaI, *d_qo, *d_beta, *d_qdo, *d_Ro, *d_Phi, *d_Fo, *d_gama, *d_alpha, *d_q, *d_qd;

	// Declare u's on device
	float *d_u1, *d_u2, *d_u3, *d_u3c, *d_u4, *d_u5, *d_u;

 // Error code to check return values for CUDA calls
    cudaError_t error;

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
    error = cudaMalloc((void **) &d_alphaI, mem_size_coef);
	if (error != cudaSuccess){
        printf("cudaMalloc d_alphaI returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_qo, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_qo returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_beta, mem_size_coef);
	if (error != cudaSuccess){
        printf("cudaMalloc d_beta returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_qdo, mem_size_q);
	if (error != cudaSuccess){
        printf("cudaMalloc d_qdo returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_Ro, mem_size_R);
	if (error != cudaSuccess){
        printf("cudaMalloc d_Ro returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_Phi, mem_size_Phi);
	if (error != cudaSuccess){
        printf("cudaMalloc d_Phi returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_Fo, mem_size_nodes);
	if (error != cudaSuccess){
        printf("cudaMalloc d_Fo returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_gama, mem_size_coef);
	if (error != cudaSuccess){
        printf("cudaMalloc d_gama returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_alpha, mem_size_coef);
	if (error != cudaSuccess){
        printf("cudaMalloc d_alpha returned error code %d, line(%d)\n", error, __LINE__);
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



// copy host memory to device

    error = cudaMemcpy(d_alphaI, h_alphaI, mem_size_coef, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_alphaI,h_alphaI) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_qo, h_qo, mem_size_q, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_qo,h_qo) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_beta, h_beta, mem_size_coef, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_beta,h_beta) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_qdo, h_qdo, mem_size_q, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_qdo,h_qdo) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_Ro, h_Ro, mem_size_R, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_Ro,h_Ro) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_Phi, h_eigenVecs, mem_size_Phi, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_Phi,h_eigenVecs) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_Fo, h_Fo, mem_size_nodes, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_Fo,h_Fo) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_gama, h_gama, mem_size_coef, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_gama,h_gama) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_alpha, h_alpha, mem_size_coef, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_alpha,h_alpha) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

/*#########---------Ends variables allocation---------------#########*/



/*#########---------Get qd---------------#########*/

	// Setup Kernel parameters
/*###---------------------------------------------####*/

/*
MatByVec?
for(i=0;i<RowsA;i++)
for(j=0;j<ColsA;j++)
sum+=A[i*RowsA+j]*B[j];
 */

/*###----------------First Part--------------------####*/ 
        // First part
		// eigenC x 1
		// u1=(alpha-Identity)*qo       Can be done with VectorAdd Kernel, say whaat?

		//Launch one extra block to make it multiple of 32
	unsigned int threadsPB= 512;
	unsigned int BPG = (eigencount + threadsPB - 1) / threadsPB;
	dim3 threadsPerBlock(threadsPB);
	dim3 blocksPerGrid(BPG);
	printf("CUDA MatByVec (u1) kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
		// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_alphaI, d_qo, d_u1, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u1) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	error = cudaMemcpy(h_q, d_u1, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
//	__syncthreads();
//--------------------Second Part-------------------	

	// Second part
	// eigenC x 1
	// u2=beta*qdo

	// Same kernel parameters as for u1
	printf("CUDA MatByVec (u2) kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
		// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_beta, d_qdo, d_u2, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	error = cudaMemcpy(h_q, d_u2, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
//	__syncthreads();

//-------------------------Third Part .1 --------------------------	

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

//	__syncthreads()

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

	error = cudaMemcpy(h_Ro, d_u3c, mem_size_Phi, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	// New kernel parameters
	unsigned int numElements = size_nodes*size_eigen;
	blocksPerGrid=(numElements + threadsPB - 1) / threadsPB;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
	// Arguments: Result, Original, # Elements in Matrix
    kMatrixTranspose<<< blocksPerGrid, threadsPerBlock>>>(d_u3,d_u3c, size_nodes, size_eigen);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch matrixTranspose kernel (u3) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	error = cudaMemcpy(h_Ro, d_u3, mem_size_Phi, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

//------------------------Third part .3-----------------------------
	// Third part .3
	// eigencount x 1
    // u4 = u3*Fo
	blocksPerGrid=(eigencount + threadsPB - 1) / threadsPB;
	printf("CUDA MatByVec (u4) kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
		// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_u3, d_Fo, d_u4, eigencount, size_nodes);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u4) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	error = cudaMemcpy(h_q, d_u4, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
//-----------------------Part 4-----------------------
	// eigencount x 1
    // u5 = gama*u4

	// Same kernel params as u4
	printf("CUDA MatByVec (u5) kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
		// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_gama, d_u4, d_u5, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u5) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	error = cudaMemcpy(h_q, d_u5, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

//--------------------------------------------
	// Fourht part sum all and divive by h
	// u2=u2+u5
	// vector + vector kernel u2=u2+u5

	// Same kernel parameters
	printf("CUDA MatByVec (u2.2) kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
		// Arguments: VectorOrig&Res, Vector2, # of Elements
		// CHECK
		// Arguments: Vector1, Vector2, VectorResult,  # of Elements
	kVectorAdd<<< blocksPerGrid, threadsPerBlock>>>(d_u2, d_u5, d_u2, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u2.2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	error = cudaMemcpy(h_q, d_u2, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	// syncthreads()


	// qd=(u1+u2)/h

	// vector + vector u1=u1+u2
	// Same kernel parameters
	printf("CUDA VectorAdd (u1.2) kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
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
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
	// Arguments: vector, scalar, # elements
    kVectorScalar<<< blocksPerGrid, threadsPerBlock>>>(d_qd, 1/h_h,  eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


	error = cudaMemcpy(h_q, d_qd, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
/*#########---------Ends Get qd---------------#########*/


/*#########---------Get q---------------#########*/

	// First part
	// eigenC x 1
	// u1=alpha*qo       Can be done with VectorAdd Kernel, say whaat?

	// Same parameters as other kernels

	printf("CUDA MatByVec (u1.q) kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
		// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_alpha, d_qo, d_u1, eigencount, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u1.q) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

	// Second part
	// eigenC x 1
	// q=u1+u2
// Same kernel parameters
	printf("CUDA VectorAdd (q) kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
		// Arguments: Vector1, Vector2, VectorResult,  # of Elements
	kVectorAdd<<< blocksPerGrid, threadsPerBlock>>>(d_u1, d_u2, d_q, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch VectorAdd kernel (u1.2) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	
/*#########---------Ends Get q---------------#########*/

/*#########---------Calculate u---------------#########*/

	//   (node_count x node_dimensions) x 1
	// u=Phi *q

// Same parameters as other kernels

	printf("CUDA MatByVec (u) kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
		// Arguments: Matrix, Vector, Result, RowsMatrix, ColsMatrix
	kMatVector<<< blocksPerGrid, threadsPerBlock>>>(d_Phi, d_q, d_u, size_nodes, eigencount);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch MatByVec kernel (u) (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

/*#########---------Ends Calculate u---------------#########*/


//Try not to delete U to be able to use it in the next kernel
	// Should I declare u  on device from main function?
	// May have to delete cudaDeviceReset(); in other functions


/*#########---------Free Memory---------------#########*/
 // Copy result from device to host
    error = cudaMemcpy(h_u, d_u, mem_size_nodes, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_u,d_u) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(h_q, d_q, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_q,d_q) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(h_qd, d_qd, mem_size_q, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_qd,d_qd) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

	cudaFree(d_u1);
	cudaFree(d_u2);
	cudaFree(d_u3);
	cudaFree(d_u4);
	cudaFree(d_u5);
	cudaFree(d_alphaI);
	cudaFree(d_qo);
	cudaFree(d_beta);
	cudaFree(d_qdo);
	cudaFree(d_Ro);
	cudaFree(d_Phi);
	cudaFree(d_Fo);
	cudaFree(d_gama);
	cudaFree(d_alpha);
	cudaFree(d_u);


/*#########---------Ends Free Memory---------------#########*/
	}
