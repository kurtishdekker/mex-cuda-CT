# technical ramblings
This section contains a non-comprehensive, unfiltered set of notes on some of the oddities of these tools.

## compilation in 2 stages
The `compile.m` script runs in 2 stages. 
First, the CUDA code is compiled by the NVIDIA `nvcc` compiler into `.o` object files, 
which are then compiled into `.mex` files in MATLAB using the `mex` command. 
This is a holdover from earlier versions of MATLAB where I had difficulty making `mex` cooperate with `nvcc` at the time. 
It is very likely that the `compile.m` script could be simplified to simply use a single-stage `mex` command.

## code duplication from CUDAmex_FP/BP to CUDAmex_oscIter.cu
The projection operators are duplicated in `CUDAmex_oscIter.cu`. This is a holdover from when I started out and did not know how to use `nvcc` and `mex` with multiple files. A housekeeping task would be to separate the projection kernels out from the `mex` function code to avoid code copy-pasting.

## projection operators and interpolation
We use interpolation for both forward and back projection operations, making use of hardware acceleration (texture memory) 
where possible, for speed.

### forward projection 
Forward projection is done by parameterizing a ray line using the classic Siddon<sup>1</sup> or Jacobs<sup>2</sup> method of 
determining the intersections of the ray with the 3D volume. 
However, instead of computing the set of ray/voxel boundary intersections, we simply use a finite step size (half of a voxel width) and 
use 3D tri-linear interpolation to calculate the contribution of voxels to that ray along each step. For forward projection, the image volume
is stored as a `tex3D` texture memory object, enabling hardware interpolation.

### backprojection 
#### fan, parallel, and cone-beam geometry
Backprojection is accomplished by a voxel-based approach. For a given voxel, we compute the coordinates of the voxel center projected 
onto the projection grid, then use a 2D bi-linear interpolation to obtain the value to backproject onto that voxel. In this case, 
the projection is stored as a `tex2D` object, enabling hardware interpolation.

#### general3D geometry
For "general 3D" geometry, we cannot use a voxel-based backprojection as the "projection" in this case is an array of measured values along
non-regularly spaced (either angularly or linearly) ray lines. Therefore, we use a ray-driven backprojector. 
Effectively, we perform the inverse of the forward projection operation described above. 
This is problematic because we are parallelized on the individual rays, and so multiple threads may be trying to write to the same voxel.
Without accounting for this, information would be lost. Fortunately, `atomicAdd` operations allow threads to "block" a voxel, forcing 
other threads to wait their turn to increment its value. While this prevents loss of information, it is significantly slower due to 
the sequential writing of many voxels. It will also lead to discretization artifacts if ray super-sampling is not used as the implicit interpolation used for the standard geometries is not used here.

## recon size set by projection / projection size set by recon
In this version of the code, by default the reconstruction dimensions (MxMxN) will be set by the projection dimensions (MxN), 
or *vice versa*. Our approach to reconstructing at a desired voxel size is to downsize the projection images (averaging pixels)
prior to reconstruction. Likely owing to our use of implicit interpolation *via* textures (see above), we have not observed
discretization artifacts using this approach. However, it would be a fairly simple modification to decouple the projection and 
reconstruction dimensions, thus allowing for ray super-sampling of voxels.
