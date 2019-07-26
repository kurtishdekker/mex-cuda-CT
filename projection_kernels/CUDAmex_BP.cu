/*CUDAmex_BP.cu
Computes backprojection operation for a parallel-beam, stacked fan-beam, or cone-beam CT geometry. 

Authors: Kurtis Dekker
         David Turnbull (prior version of cone beam backprojector)

Created : July 2 2015
Modified: March 21 2016 - support array of projection angles (non-equal spacing)
		        July 23 2019  - cleanup and commenting for public release
*/

#include "mex.h"
 
// This define command is added for the M_PI constant
#define _USE_MATH_DEFINES 1
#include <math.h>
#include <cuda.h>

void checkCudaError(const char *msg);
// 2D float texture
texture<float, 2, cudaReadModeElementType> projtex; //for storing a projection

__global__ void backprojKernel_par(float* output, float v0, float sinBeta, float cosBeta, float SR, unsigned int width, unsigned int height)
{
    /*
        inputs:
            float* output           - pointer to 3D FP device array to contain recon volume
            float v0                - centrepoint (vertical direction) on the detector
            float sinBeta,cosBeta   - sin and cosine of the projection angle Beta
            float SR                - Source to Rotation axis distance in pixel units
            unsigned int width      - width (pixels) of the reconstruction volume and projection
            unsigned int height     - height (pixels) of the recon volume / projection data

        ouputs:
            function has no return value but fills the variable 'output' with backprojected image

        notes:
            The size (x,y) of the projection images must match the reconstruction dimensions. that is, a 640(w) by 480(h) projection
            will result in a reconstruction of 640x640x480slices. The voxel size in the reconstruction will be equal to the pixel size
            in the projection images at the rotation axis.

            TODO: modify for 3D // is easily done by simply allowing 'SR' to be set as Inf, then enforcing that 'U' = 1 in backprojection.
                  This is a simple modification if telecentric lens system becomes the desired approach.

    */

    //calculate x and y indices from thread/block index. These indices define the <x,y> position inside the recon volume (x,y) as seen from top
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //calculate the radius of the reconstruction circle. Outside the "circle of reconstruction", no useful data exists-> should be masked out after
    unsigned int halfwidth = width/2 - 1;

    //only backproject to those voxels which lie within the "circle of reconstruction" which is defined by the size of the projection.
    //so the resolution of the reconstruction (voxel dim) should be set to equal that of the image pixels at the axis of rotation (can compute this from the magnification)
    if( ((ix-halfwidth)*(ix-halfwidth) + (iy-halfwidth)*(iy-halfwidth)) < halfwidth*halfwidth) //inside circle of reconstruction
    {
        float x = ix - (halfwidth + 0.5f);
        float y = iy - (halfwidth + 0.5f);

        //float sinBeta,cosBeta;
        //sincos(beta,&sinBeta,&cosBeta);

        //compute rotated coordinates <s,t> for projection at angle "Beta"
        float s = x*cosBeta + y*sinBeta;
        float t = -x*sinBeta + y*cosBeta;

        //compute "U", which is the scaling factor needed to convert from image voxel coordinates to projection coordinates
        //the logic here is simply based on similar triangles which have the optic axis as their base.
        //float U = SR / (SR-s);
        float U = 1.f;
        //compute horizontal (y) and vertical (z) coordinates in the projection
        float proj_y= t*U + halfwidth + 1.f;
        float proj_z= -v0*U + v0 + 0.5f;

        //loop over 'z' in image space (vertical direction) and backproject pixel value
        for ( unsigned int iz = 0; iz < height; iz++)
        {
            output[ix + width*iy + width*width*iz] += U*U * tex2D(projtex,proj_y,proj_z);
            proj_z += U;
        }
    }
}

__global__ void backprojKernel_fan(float* output, float v0, float sinBeta, float cosBeta, float SR, unsigned int width, unsigned int height)
{
    /*
        inputs:
            float* output           - pointer to 3D FP device array to contain recon volume
            float v0                - centrepoint (vertical direction) on the detector
            float sinBeta,cosBeta   - sin and cosine of the projection angle Beta
            float SR                - Source to Rotation axis distance in pixel units
            unsigned int width      - width (pixels) of the reconstruction volume and projection
            unsigned int height     - height (pixels) of the recon volume / projection data

        ouputs:
            function has no return value but fills the variable 'output' with backprojected image

        notes:
            The size (x,y) of the projection images must match the reconstruction dimensions. that is, a 640(w) by 480(h) projection
            will result in a reconstruction of 640x640x480slices. The voxel size in the reconstruction will be equal to the pixel size
            in the projection images at the rotation axis.
    */

    //calculate x and y indices from thread/block index. These indices define the <x,y> position inside the recon volume (x,y) as seen from top
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //calculate the radius of the reconstruction circle. Outside the "circle of reconstruction", no useful data exists-> should be masked out after
    unsigned int halfwidth = width/2 - 1;

    //only backproject to those voxels which lie within the "circle of reconstruction" which is defined by the size of the projection.
    //so the resolution of the reconstruction (voxel dim) should be set to equal that of the image pixels at the axis of rotation (can compute this from the magnification)
    if( ((ix-halfwidth)*(ix-halfwidth) + (iy-halfwidth)*(iy-halfwidth)) < halfwidth*halfwidth) //inside circle of reconstruction
    {
        float x = ix - (halfwidth + 0.5f);
        float y = iy - (halfwidth + 0.5f);

        //float sinBeta,cosBeta;
        //sincos(beta,&sinBeta,&cosBeta);

        //compute rotated coordinates <s,t> for projection at angle "Beta"
        float s = x*cosBeta + y*sinBeta;
        float t = -x*sinBeta + y*cosBeta;

        //compute "U", which is the scaling factor needed to convert from image voxel coordinates to projection coordinates
        //the logic here is simply based on similar triangles which have the optic axis as their base.
        float U = SR / (SR-s);

        //compute horizontal (y) and vertical (z) coordinates in the projection
        float proj_y= t*U + halfwidth + 1.f;
        //float proj_z= -v0*U + v0 + 0.5f;
        float proj_z = 0.5f;
        //loop over 'z' in image space (vertical direction) and backproject pixel value
        for ( unsigned int iz = 0; iz < height; iz++)
        {
            output[ix + width*iy + width*width*iz] += U*U * tex2D(projtex,proj_y,proj_z);
            proj_z += 1;
        }
    }
}

__global__ void backprojKernel_cone(float* output, float v0, float sinBeta, float cosBeta, float SR, unsigned int width, unsigned int height)
{
    /*
        inputs:
            float* output           - pointer to 3D FP device array to contain recon volume
            float v0                - centrepoint (vertical direction) on the detector
            float sinBeta,cosBeta   - sin and cosine of the projection angle Beta
            float SR                - Source to Rotation axis distance in pixel units
            unsigned int width      - width (pixels) of the reconstruction volume and projection
            unsigned int height     - height (pixels) of the recon volume / projection data

        ouputs:
            function has no return value but fills the variable 'output' with backprojected image

        notes:
            The size (x,y) of the projection images must match the reconstruction dimensions. that is, a 640(w) by 480(h) projection
            will result in a reconstruction of 640x640x480slices. The voxel size in the reconstruction will be equal to the pixel size
            in the projection images at the rotation axis.

            TODO: modify for 3D // is easily done by simply allowing 'SR' to be set as Inf, then enforcing that 'U' = 1 in backprojection.
                  This is a simple modification if telecentric lens system becomes the desired approach.

    */

    //calculate x and y indices from thread/block index. These indices define the <x,y> position inside the recon volume (x,y) as seen from top
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //calculate the radius of the reconstruction circle. Outside the "circle of reconstruction", no useful data exists-> should be masked out after
    unsigned int halfwidth = width/2 - 1;

    //only backproject to those voxels which lie within the "circle of reconstruction" which is defined by the size of the projection.
    //so the resolution of the reconstruction (voxel dim) should be set to equal that of the image pixels at the axis of rotation (can compute this from the magnification)
    if( ((ix-halfwidth)*(ix-halfwidth) + (iy-halfwidth)*(iy-halfwidth)) < halfwidth*halfwidth) //inside circle of reconstruction
    {
        float x = ix - (halfwidth + 0.5f);
        float y = iy - (halfwidth + 0.5f);

        //compute rotated coordinates <s,t> for projection at angle "Beta"
        float s = x*cosBeta + y*sinBeta;
        float t = -x*sinBeta + y*cosBeta;

        //compute "U", which is the scaling factor needed to convert from image voxel coordinates to projection coordinates
        //the logic here is simply based on similar triangles with the optic axis as their base.
        float U = SR / (SR-s);

        //compute horizontal (y) and vertical (z) coordinates in the projection
        float proj_y= t*U + halfwidth + 1.f;
        float proj_z= -v0*U + v0 + 0.5f;

        //loop over 'z' in image space (vertical direction) and backproject pixel value
        for ( unsigned int iz = 0; iz < height; iz++)
        {
            output[ix + width*iy + width*width*iz] += U*U * tex2D(projtex,proj_y,proj_z);
            proj_z += U;
        }
    }
}
 
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] ) 
{
   if (nrhs < 2)
     mexErrMsgIdAndTxt("mex_conebeam_fdk:invalidInput", "3 arguments expected");

   float *P;
   const mwSize *dim_array;         
   P = (float *)mxGetData(prhs[0]);
   float *projAngs;
   projAngs = (float *)mxGetData(prhs[2]);
   dim_array=mxGetDimensions(prhs[0]);

   float D = mxGetScalar(prhs[1]);
  // float projSpacing = mxGetScalar(prhs[2]);
   float beta;

   int geomFlag = mxGetScalar(prhs[3]);

   // reconstruction volume
   float *recon;
   const mwSize dims[]={dim_array[0], dim_array[0], dim_array[1]};
   plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
   recon = (float*)mxGetData(plhs[0]);

   // u,v coordinates of the center of the detector plane
   float u0 = float(dim_array[0]-1)/2;
   float v0 = float(dim_array[1]-1)/2;

   unsigned int width = dim_array[0];
   unsigned int height = dim_array[1];


   // Allocate recon in device memory
   float* output;
   size_t outputMemSize = width * width * sizeof(float)*(height);
   cudaMalloc((void**)&output, outputMemSize);
   checkCudaError("output malloc");
   cudaMemset(output,0,outputMemSize);
         
   // allocate texture memory on GPU for a projection
   size_t sizeTex = width*height*sizeof(float);
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
   cudaArray *cuArray1;
   
   cudaMallocArray(&cuArray1, &channelDesc, width, height);
   checkCudaError("texture 1 malloc");
   cudaBindTextureToArray(projtex, cuArray1, channelDesc);
   projtex.filterMode = cudaFilterModeLinear;
   projtex.normalized = false;
   projtex.addressMode[0] = cudaAddressModeClamp;
   projtex.addressMode[1] = cudaAddressModeClamp;

   // these are 'temporarily' hardwired for recon of 512*512
   dim3 dimBlock(32,32);
   int nblocks;
   if ( width % 32 == 0){
       nblocks = width/32;
   } else {
       nblocks = width/32 + 1;
   }
   //mexPrintf("nblocks = %d",nblocks);
   dim3 dimGrid(nblocks,nblocks);

   float *ptrRecon;
   size_t projOffset = dim_array[0]*dim_array[1];  // offset to each projection
   float sinBeta, cosBeta;

   
   for(size_t projection=0; projection<dim_array[2]; projection+=1)   // hardwired for 512 projections
   {
      beta = projAngs[projection];
      sincosf(beta, &sinBeta, &cosBeta);
      cudaMemcpyToArray(cuArray1, 0, 0, &P[  projection*projOffset  ], sizeTex, cudaMemcpyHostToDevice);
      
      switch (geomFlag)
      {
          case 0:
            backprojKernel_par<<<dimGrid,dimBlock>>>( output, v0, sinBeta, cosBeta, D, width, height);
            break;
          case 1:
            backprojKernel_fan<<<dimGrid,dimBlock>>>( output, v0, sinBeta, cosBeta, D, width, height);
            break;
          case 2: 
            backprojKernel_cone<<<dimGrid,dimBlock>>>( output, v0, sinBeta, cosBeta, D, width, height);
            break;
          default:
            mexErrMsgTxt("invalid geometry");
      }
      
      cudaThreadSynchronize();
      checkCudaError("main kernel invocation");
   } 
   
   ptrRecon = recon;
   cudaMemcpy(ptrRecon, output, outputMemSize, cudaMemcpyDeviceToHost);
   checkCudaError("output error");
  
   cudaFree(output);
   cudaUnbindTexture(projtex);
   cudaFreeArray(cuArray1); 
   return;
}

//check for CUDA errors
void checkCudaError(const char *msg)
{
   cudaError_t err = cudaGetLastError();
   if( cudaSuccess != err )
   {
      mexErrMsgTxt(msg);
   }
}
