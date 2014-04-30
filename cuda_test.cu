// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void kern(int i)
{

i=i+i;
}


//__global__ void kernel(float scale)
__global__ void kernel(float *g_nodes,float scale)
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

	g_nodes[BDIM*BID+TID]=g_nodes[BDIM*BID+TID]*scale;
//	scale*scale;
}
__global__ void kernelRemaining(float *g_nodes,float scale, unsigned int threadsDone)
{
    // write data to global memory
    const unsigned int TID = threadIdx.x;

	g_nodes[threadsDone-1+TID]=g_nodes[threadsDone-1+TID]*scale;
}

//float *cuda_data=NULL;
extern "C" void map_texture(void *cuda_dat, size_t siz,cudaGraphicsResource *resource)
{
size_t size;
cudaGraphicsResourceGetMappedPointer((void **)(&cuda_dat), &size, resource);
}

extern "C" bool
runTest(const int argc, const char **argv, float *buffer, int node_count, float scale)
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
    kernel<<< grid, threads >>>(buffer,scale);
if(totalThreads>512)
	kernelRemaining<<< 1,mod >>>(buffer,scale, threadsDone);
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
}
