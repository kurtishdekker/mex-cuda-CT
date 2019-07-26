/*CUDAmex_TVmin_3D.cu

Perform total variation minimization de-noising on a 3D image volume. 
Compiles to a MEX function with MATLAB.

Usage (matlab):
    	image_out = CUDAmex_tvmin_3D(image_in,tvConst)

TODO:
	remove hard-coded number of TV-minimization iterations, allow specification in function call.


References:
[1] K. H. Dekker, J. J. Battista, and K. J. Jordan, Medical Physics, vol. 44, no. 12, pp. 6678–6689, Dec. 2017.
[2] D. Matenine, Y. Goussard, and P. Després, Medical Physics, vol. 42, no. 4, pp. 1505–1517, Apr. 2015.


Author   : Kurtis H Dekker, PhD
Created  : April 10 2017
Modified : July 23, 2019

*/


//INCLUDES
#include "mex.h"
 
// This define command is added for the M_PI constant
#define _USE_MATH_DEFINES 1
#include <math.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <algorithm>
#include <cstdlib>


//DEFINES

// macro to handle CUDA errors. Wrap CUDA mallocs and memCpy in here
#define CHECK_CUDA_ERROR(x) do {\
	cudaError_t res = (x); \
	if (res != cudaSuccess) { \
    sprintf(s,"CUDA ERROR: %s = %d (%s) at (%s:%d)\n",#x, res, cudaGetErrorString(res),__FILE__,__LINE__);\
	mexErrMsgTxt(s);\
	\
	} \
} while(0)


//TEMPLATES AND STRUCTS (for thrust operations)
// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

//single precision a*x + y.
// y <- y + a*x
struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};


//FUNCTION PROTOTYPES
__global__ void tvGradientKernel(float* output, float* input, unsigned int Nx, unsigned int Ny, unsigned int Nz); //device TV grad kernel


/* END OF PREAMBLE */



__global__ void tvGradientKernel(float* output, float* input, unsigned int Nx, unsigned int Ny, unsigned int Nz)
{
    /* 
        inputs:
            float* output           - pointer to 3D FP device array containing the output array
            float* input            - pointer to 3D FP device array containing the input array
            unsigned int Nx,Ny,Nz   - length/width/height of 3d array, for bounds checking
        
        outputs:
            function has no return value, but fills "output" with the local values of the TV gradient

        notes:
        
    */

    //calculate x,y,z indices from thread/block index. these indices define position within the volume
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (( ix < (Nx-1) && iy < (Ny-1) && iz < (Nz-1)) && (ix >= 1 && iy >=1 && iz >= 1)) //check bounds
    {
        //define variables
        float eps = 1.0e-6;
        float xyz, xmyz, xpyz, xymz, xypz, xyzm, xyzp, xmypz, xpymz, xymzp, xypzm, xpyzm, xmyzp;

        //for simplicity of coding the actual calculations, grab all the relevant values here
        //todo: examine what happens to this on compile.
        //todo: look @ using macros instead of variables here? is it better/worse/equivalent?
        xyz =   input[ix + Nx*iy + Nx*Ny*iz];
        xmyz =  input[(ix-1) + Nx*iy + Nx*Ny*iz];
        xpyz =  input[(ix+1) + Nx*iy + Nx*Ny*iz];
        xymz =  input[ix + Nx*(iy-1)+ Nx*Ny*iz];
        xypz =  input[ix + Nx*(iy+1)+ Nx*Ny*iz];
        xyzm =  input[ix + Nx*iy + Nx*Ny*(iz-1)];
        xyzp =  input[ix + Nx*iy + Nx*Ny*(iz+1)];
        xmypz = input[(ix-1) + Nx*(iy+1)+ Nx*Ny*iz];
        xpymz = input[(ix+1)+ Nx*(iy-1)+ Nx*Ny*iz];
        xymzp = input[ix + Nx*(iy-1)+ Nx*Ny*(iz+1)];
        xypzm = input[ix + Nx*(iy+1) + Nx*Ny*(iz-1)];
        xmyzp = input[(ix-1)+ Nx*iy + Nx*Ny*(iz+1)];
        xpyzm = input[(ix+1) + Nx*iy + Nx*Ny*(iz-1)];

    //calculate the TV gradient. This could be condensed to one step but it's easier to read / debug here.
        float tval;
        tval = ((xyz - xmyz) + (xyz - xymz) + (xyz - xyzm))/(eps + sqrt((xyz-xmyz)*(xyz-xmyz) + (xyz-xymz)*(xyz-xymz) + (xyz-xyzm)*(xyz-xyzm)));

        tval -= ((xpyz - xyz))/(eps + sqrt((xpyz-xyz)*(xpyz-xyz) + (xpyz - xpymz)*(xpyz-xpymz) + (xpyz - xpyzm)*(xpyz - xpyzm)));
        
        tval -= ((xypz - xyz))/(eps + sqrt((xypz-xyz)*(xypz-xyz) + (xypz - xmypz)*(xypz-xmypz) + (xypz - xypzm)*(xypz - xypzm)));
        
        tval -= ((xyzp - xyz))/(eps + sqrt((xyzp-xyz)*(xyzp-xyz) + (xyzp - xmyzp)*(xyzp-xmyzp) + (xyzp - xymzp)*(xyzp - xymzp)));

        output[ix+Nx*iy + Nx*Ny*iz] = tval;
    }
    

}


// the mex function wrapper
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    
    if (nrhs != 2)
        mexErrMsgIdAndTxt("mex_conebeam_FP:invalidInput","2 arguments expected");

    char s [200]; //char array for error messages from CUDA_CHECK_ERROR macro

    //input image volume
    float *array;
    array = (float *)mxGetData(prhs[0]);
    const mwSize *dim_array;
    dim_array = mxGetDimensions(prhs[0]);
    
    int numel = (dim_array[0]*dim_array[1]*dim_array[2]); //number of elements in volume

    float tv_const = mxGetScalar(prhs[1]); //TV-reg constant 'c'


    //array to store TV-gradient
    float *outArray;
    const mwSize dims[]={dim_array[0],dim_array[1],dim_array[2]};
    plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    outArray = (float*)mxGetData(plhs[0]);

    //allocate TV gradient volume in device memory
    float *output;
    size_t outputMemSize = dim_array[0] * dim_array[1] * sizeof(float) * dim_array[2];
    CHECK_CUDA_ERROR(cudaMalloc((void**)&output, outputMemSize));

    //allocate input image volume in device memory
    //NOTE: need to allocate both because if we update on the image directly it impacts other kernels, no bueno.
    float *input;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&input, outputMemSize));
    CHECK_CUDA_ERROR(cudaMemcpy(input, array, outputMemSize, cudaMemcpyHostToDevice));

    //TODO: vary blocksize to see if efficiency improves
    dim3 dimBlock(32,32,1);
    int nblocksX, nblocksY, nblocksZ;
   if ( dim_array[0] % 32 == 0){
       nblocksX = dim_array[0]/32;
   } else {
       nblocksX = dim_array[0]/32 + 1;
   }
   
   if ( dim_array[1] % 32 == 0){
       nblocksY = dim_array[1]/32;
   } else {
       nblocksY = dim_array[1]/32 + 1;
   }

   if ( dim_array[2] % 1 == 0){
       nblocksZ = dim_array[2]/1;
   } else {
       nblocksZ = dim_array[2]/1 + 1;
   } 

   dim3 dimGrid(nblocksX,nblocksY,nblocksZ);
    
   for (int i = 0; i<20; i++) //currently, use 20 iterations of TV-minimization gradient descent. 
   {
        tvGradientKernel<<<dimGrid,dimBlock>>>(output, input, dim_array[0],dim_array[1],dim_array[2]);
        
        thrust::device_ptr<float> oPtr = thrust::device_pointer_cast(output); //pointer to output, containing the TV gradient
        thrust::device_ptr<float> iPtr = thrust::device_pointer_cast(input); //pointer to input, containing the image volume
       
        //calculate sum of squared TV gradient values, here we're using a transform-reduce in thrust libraries
        float magnitude = std::sqrt(thrust::transform_reduce(oPtr,oPtr+numel,square<float>(),0.0, thrust::plus<float>())); 
        
        //calculate the new image after TV-min step, this is formulated as a linearized equation a*x+y, so use a SAXPY
        thrust::transform(oPtr, oPtr+numel,iPtr,iPtr, saxpy_functor(-tv_const/magnitude));
        
   } 

   //point to and copy memory from the TV-denoised image on GPU to the output array for the mex file
   float* ptrArray;
   ptrArray = outArray;
   CHECK_CUDA_ERROR(cudaMemcpy(ptrArray, input, outputMemSize, cudaMemcpyDeviceToHost));

   //free up GPU memory
   cudaFree(output);
   cudaFree(input);
   return;

}   
    
