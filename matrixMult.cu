#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "cuPrintf.cu"

template <int BLOCK_SIZE> __global__ void kmatrixMult(float *A, float *B, float *C,int ARows, int ACols, int BRows, int BCols)
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

/**
 * Function to call from main C++ code
 */
extern "C" bool MatrixMult (float *h_A, float *h_B, float *h_C ,int RowsA, int ColsA, int RowsB, int ColsB, const int block_size)
{
	// Allocate matrices on GPU
/*###---------------------------------------------####*/

	// Get sizes from system memory pointers
	unsigned int size_A = RowsA * ColsA;
	unsigned int size_B = RowsA * ColsB;
	unsigned int mem_size_A, mem_size_B, mem_size_C;
	mem_size_A = sizeof(float) * size_A;
	mem_size_B = sizeof(float) * size_B;
	mem_size_C = sizeof(float) * RowsA*ColsB;

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Error code to check return values for CUDA calls
    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }


/*###---------------------------------------------####*/

	// Setup Kernel parameters
/*###---------------------------------------------####*/
	// ThreadsPerBlock, 2 dimensions 16*16
	dim3 threads(block_size, block_size);

	//Launch one extra block to make it multiple of 32
    dim3 grid((ColsB+threads.x-1) / threads.x, (RowsA+threads.y-1) / threads.y);

/*###---------------------------------------------####*/


	// Execute kernel
/*###---------------------------------------------####*/
	printf("\nComputing result using CUDA...\n");
    printf("ThreadsPerBlock:%u,%u   BlocksPerGrid:%u,%u\n",threads.x,threads.y,grid.x,grid.y);
	// Init cuda Printf In case its needed
    cudaPrintfInit ();


//	cudaCall = VectorAdd (h_A,h_B,numElements);

	// Exec kernel
	kmatrixMult<16><<< grid, threads >>>(d_A, d_B, d_C, RowsA,ColsA, RowsB, ColsB);


	printf("GPU is done computing\n");

	// CudaPrintf Stuff
	cudaPrintfDisplay (stdout, true);
    cudaPrintfEnd ();


    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

/*###---------------------------------------------####*/

	// Free memory on GPU
/*###---------------------------------------------####*/
   // Free device global memory
    error = cudaFree(d_A);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix A (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_B);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix B (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    error = cudaFree(d_C);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix C (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
/*###---------------------------------------------####*/

 
    // Reset the device and exit
    error = cudaDeviceReset();

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
		return false;
    }

    printf("Done\n");
    return true;
}

