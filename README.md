mwgpu
=====

Modal Warping Using Parallel Programming on the GPU

#######################
OpenGL on CUDA v 1.0

- Zoom is performed by scaling node locations by some factor
- OpenGL is using a VBO
- Calculations are done on the GPU
- CUDA calculations are applied directly to the VBO avoiding memory transfer
- Thread/Block/Grid are computed "optimally" 

#######################


#######################
CUDA Displacement v 1.1

- Loading values of modified K and M Matrices and fixed_nodes indices
- Only one function to calculate and render
- Render using original nodes and not incrementally (WORKS!!!)
- Zoom stopped working due above
- Change colors of object
- Still have to check DrawElements to see if size is correct


#######################

#######################
Modal Warping on GPU v 1.2

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
Modal Warping on GPU v 1.3.1
- R computation works serially
- Multiple Fragment Shaders
- Changed primitive types (Draw)
- Started working on R in parallel


#######################

TODO:
OpenGL:
- Render Info Text
- Click to fix face functionality?***

CUDA:
- Build R matrix?
- Include R matrix in computation
- Declare globals on device for constant arrays
- Add Eigenvectors computation***
- R in parallel:
	each thread computes skew matrix for every node
-Cleanup Again


C++:
- Check variables not used
- Make it modular (File_Loading.h, etc)

########################


