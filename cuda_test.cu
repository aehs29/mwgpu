// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>



__global__ void kernel(float *g_nodes)
{
    // write data to global memory
    const unsigned int tid = threadIdx.x;
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

	g_nodes[tid]=g_nodes[tid]*1.2;
}


extern "C" bool
runTest(const int argc, const char **argv, float *nodes, int node_count)
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

    const unsigned int num_threads = node_count*3;
//    assert(0 == (len % 4));
    const unsigned int mem_size = sizeof(float) * node_count*3;
//    const unsigned int mem_size_int2 = sizeof(int2) * len;

    // allocate device memory
    float *d_nodes;
    checkCudaErrors(cudaMalloc((void **) &d_nodes, mem_size));
    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_nodes, nodes, mem_size,
                               cudaMemcpyHostToDevice));
    // allocate device memory for int2 version
//    int2 *d_data_int2;
//    checkCudaErrors(cudaMalloc((void **) &d_data_int2, mem_size_int2));
    // copy host memory to device
 //   checkCudaErrors(cudaMemcpy(d_data_int2, data_int2, mem_size_int2,
  //                             cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3 grid(1, 1, 1);
    dim3 threads(num_threads, 1, 1);
    //dim3 threads2(len, 1, 1); // more threads needed fir separate int2 version
    // execute the kernel
    kernel<<< grid, threads >>>((float*) d_nodes);
    //kernel2<<< grid, threads2 >>>(d_data_int2);

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // compute reference solutions
//    char *reference = (char *) malloc(mem_size);
//    computeGold(reference, data, len);
//    int2 *reference2 = (int2 *) malloc(mem_size_int2);
//    computeGold2(reference2, data_int2, len);

    // copy results from device to host
    checkCudaErrors(cudaMemcpy(nodes, d_nodes, mem_size,
                               cudaMemcpyDeviceToHost));
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
    checkCudaErrors(cudaFree(d_nodes));
//    checkCudaErrors(cudaFree(d_data_int2));
//    free(reference);
//    free(reference2);

    return success;
}
