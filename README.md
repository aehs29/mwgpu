mwgpu
=====

Modal Warping Using Parallel Programming on the GPU

#######################
OpenGL on CUDA v1.0

- Zoom is performed by scaling node locations by some factor
- OpenGL is using a VBO
- Calculations are done on the GPU
- CUDA calculations are applied directly to the VBO avoiding memory transfer
- Thread/Block/Grid are computed "optimally" 

#######################


#######################
CUDA Displacement v1.1

- Loading values of modified K and M Matrices and fixed_nodes indices
- Only one function to calculate and render
- Render using original nodes and not incrementally (WORKS!!!)
- Zoom stopped working due above
- Change colors of object
- Still have to check DrawElements to see if size is correct


#######################

#######################
Modal Warping on GPU v1.2

- Added GLM Library
- New Rotation Method
- Perspective now works correctly
- Added "Dragging" functionality
- Shaders now work correctly
- Can now change force and direction using F(1-5) keys
- Fixed EigenVectors (Phi) size
- Added Load Psy funcitonality
- Cleaned up code
- Added performance measurement (TimePerFrame)
- Fixed serial code

#######################

#######################
Modal Warping on GPU v1.3.1
- R computation works serially
- Multiple Fragment Shaders
- Changed primitive types (Draw)
- Started working on R in parallel
- Added only lines feature

#######################

#######################
Modal Warping on GPU v1.3.2
- R computation works in parallel
- Insert zeros now after computing R
- Fixed some problem when destroying shaders
- Added option to run in parallel from terminal

#######################


#######################
Modal Warping on GPU v1.3.3
- Code is now modular "Programacion modular chavoos"
- Created structs for nodes and elements
- Object now rotates around its center
- Changed filenames to mwgpu

#######################

#######################
Modal Warping on GPU v1.3.4
- Improved performance on GPU by minimizing memory transfers
- Globals declared on device at the start
- Change # of GPU threads from terminal
- Cleaned up some code

#######################


#######################
Modal Warping on GPU v1.3.5
- Simulation now on IdleFunction
- Separate performance measurement for simulation and rendering
- Added performance measurement on CUDA and C++
#######################

TODO:

OpenGL:
- Render Info Text
- Click to fix face functionality?***

CUDA:
- Add Eigenvectors computation***
- Modify kInsertZeros
  - Either keep an array with indexes to modify
  	-Sparse (Keep indexes) or Dense
  - Or use syncthreads and launch only one kernel
- Should I use SharedMem instead of GlobalMem?
- Change float64 to float32
- Coalesced memory comments
- GpuTimer timer;
  - timer.Start();
  - timer.Stop();
- Use streams for kernel launches

C++:
- GetFaces()
  - Get max and min for every axis
  - Count # nodes per face -- hashmap?
  - if node has a min or max in any axis is part of face
  -	Prioritize the face with less nodes
  - Only render nodes on faces
  - Color by face
  	- How to let the vertex know its a face
	  - VertexAttribute? containing values
	  - Check which face it is on Fragment Shader and colorize
########################


