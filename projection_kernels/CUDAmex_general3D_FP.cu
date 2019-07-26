/*CUDAmex_general3D_FP.cu

Computes a forward projection operation for a "general" geometry CT dataset wherein projection ray paths are specified by arrays of start and end points rather than a typical SAD/SDD specification.

Usage (Matlab):
	fp = CUDAmex_general3D_FP(ph,angles,pts,nProjections)
	
Inputs:
	projections        -		array of 3D image data
	angles             -		array containing the projection angles in radians
	pts                -        MxNx6 array of points defining primary ray paths through the 3D volume. 
                                The order of coordinates along dimension 3 is (x1,y1,z1,x2,y2,z2). Units are in 3D reconstruction volume voxels
    nProjections	   -		number of projections
	
Outputs:
	fp 			       -		the MxNxNprojections array of simulated projection data
	
Dependencies:
	CUDA toolkit v6.0 or later
		
NOTES:
	uses single precision. input arrays in matlab should be cast as type single

Author  : Kurtis H Dekker, PhD
Created  : April 10 2017
Modified : July 23, 2019

*/

//INCLUDES
#include "mex.h"
#include <math.h>
#include <cuda.h>


//DEFINES
#define _USE_MATH_DEFINES 1 // This define command is added for the M_PI constant
// define a NaN value and an Inf value that cannot be compiler-optimized away
#define CUDART_NAN_F __int_as_float(0x7fffffff)
#define CUDART_INF_F __int_as_float(0x7f800000)

//Texture Objects
texture<float, 3, cudaReadModeElementType> reconTex; //for storing a volume

// macro to handle CUDA errors. Wrap CUDA mallocs and memCpy in here
#define CHECK_CUDA_ERROR(x) do {\
	cudaError_t res = (x); \
	if (res != cudaSuccess) { \
    sprintf(s,"CUDA ERROR: %s = %d (%s) at (%s:%d)\n",#x, res, cudaGetErrorString(res),__FILE__,__LINE__);\
	mexErrMsgTxt(s);\
	\
	} \
} while(0)


//FUNCTION DEFINITIONS
void checkCudaError(const char *msg);

__global__ void forwardProjKernel( float* projectionOut, float* ptsArray, float bx, float by, float bz, float sinBeta, float cosBeta, float voxelSize, unsigned int imSizeX, unsigned int imSizeY, unsigned int imsizeZ, unsigned int width, unsigned int height )
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) return; //don't do anything if we're outside the projection


    //calculate ray start/end points in 3d space xyz, origin at centre of recon volume
    float3 startpoint, endpoint, currentpoint;
    int npts = width*height;
  
    startpoint.x = (ptsArray[iy * width + ix])*cosBeta - (ptsArray[iy*width+ix+npts])*sinBeta;
    startpoint.y = (ptsArray[iy * width + ix])*sinBeta + (ptsArray[iy*width+ix+npts])*cosBeta;
    startpoint.z = (ptsArray[iy * width + ix + 2*npts]);

    endpoint.x = (ptsArray[iy * width + ix + 3*npts])*cosBeta - (ptsArray[iy*width+ix + 4*npts])*sinBeta;
    endpoint.y = (ptsArray[iy * width + ix + 3*npts])*sinBeta + (ptsArray[iy*width+ix + 4*npts])*cosBeta;
    endpoint.z = (ptsArray[iy * width + ix + 5*npts]);

    float dConv = sqrtf((endpoint.x - startpoint.x)*(endpoint.x-startpoint.x) + (endpoint.y - startpoint.y)*(endpoint.y-startpoint.y) + (endpoint.z - startpoint.z)*(endpoint.z-startpoint.z));

    //calculate alphaOne values (x,y,z)
    float alphaXone, alphaXtwo, alphaYone, alphaYtwo, alphaZone, alphaZtwo;

    if (fabsf(endpoint.x - startpoint.x) >= 1e-5) {
        alphaXone = (bx - startpoint.x) / (endpoint.x - startpoint.x);
        alphaXtwo = (-bx - startpoint.x) / (endpoint.x - startpoint.x);
    } else {
        alphaXone = CUDART_NAN_F;
        alphaXtwo = CUDART_NAN_F;
    }

    if (fabsf(endpoint.y - startpoint.y) >= 1e-5) {
        alphaYone = (by - startpoint.y) / (endpoint.y - startpoint.y);
        alphaYtwo = (-by - startpoint.y) / (endpoint.y - startpoint.y);
    } else {
        alphaYone = CUDART_NAN_F;
        alphaYtwo = CUDART_NAN_F;
    }

    if (fabsf(endpoint.z - startpoint.z) >= 1e-5) {
        alphaZone = (bz - startpoint.z) / (endpoint.z - startpoint.z);
        alphaZtwo = (-bz - startpoint.z) / (endpoint.z - startpoint.z);
    } else {
        alphaZone = CUDART_NAN_F;
        alphaZtwo = CUDART_NAN_F;
    }

    float alphaXmin = fminf(alphaXone,alphaXtwo);
    float alphaXmax = fmaxf(alphaXone, alphaXtwo);
    float alphaYmin = fminf(alphaYone,alphaYtwo);
    float alphaYmax = fmaxf(alphaYone, alphaYtwo);
    float alphaZmin = fminf(alphaZone,alphaZtwo);
    float alphaZmax = fmaxf(alphaZone, alphaZtwo);

    float tmp = fmaxf(alphaXmin,fmaxf(alphaYmin, alphaZmin));
    float alphaMin = fmaxf(0.f,tmp);
    tmp = fminf(alphaXmax, fminf(alphaYmax, alphaZmax));
    float alphaMax = fminf(1.f,tmp);

    currentpoint.x = startpoint.x + alphaMin * (endpoint.x - startpoint.x);
    currentpoint.y = startpoint.y + alphaMin * (endpoint.y - startpoint.y);
    currentpoint.z = startpoint.z + alphaMin * (endpoint.z - startpoint.z);
    
    endpoint.x = startpoint.x + alphaMax * (endpoint.x - startpoint.x);
    endpoint.y = startpoint.y + alphaMax * (endpoint.y - startpoint.y);
    endpoint.z = startpoint.z + alphaMax * (endpoint.z - startpoint.z);
    
    float distance = sqrt((endpoint.x-currentpoint.x)*(endpoint.x-currentpoint.x) + (endpoint.y-currentpoint.y)*(endpoint.y-currentpoint.y) + (endpoint.z-currentpoint.z)*(endpoint.z-currentpoint.z));

    float travelled = 0.f;
    
    float stepsize = voxelSize / 2.0f;

    float stepsizeX = (endpoint.x - currentpoint.x) / distance * stepsize;
    float stepsizeY = (endpoint.y - currentpoint.y) / distance * stepsize;
    float stepsizeZ = (endpoint.z - currentpoint.z) / distance * stepsize;

    float dval=0.f;
    //loop
    while(travelled < distance) {
        dval += tex3D(reconTex, currentpoint.x - bx, currentpoint.y - by, currentpoint.z - bz) * stepsize;
        currentpoint.x += stepsizeX;
        currentpoint.y += stepsizeY;
        currentpoint.z += stepsizeZ;
        travelled += stepsize;
    }

    projectionOut[iy * width + ix] = dval;
    
}

// the mex function wrapper
//CUDAmex_forwardProjection(image, D, projAngles, singleProjPoints, nProj)
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("mex_conebeam_FP:invalidInput","4 arguments expected");
  
    //get mex arguments
    float* recon; recon = (float *)mxGetData(prhs[0]); //image data
    float* projAngs; projAngs = (float *)mxGetData(prhs[1]); //projection angles
    float* h_ptsArray; h_ptsArray = (float *)mxGetData(prhs[2]); //points array (mxnx6)
    int nProj = mxGetScalar(prhs[3]);

    //get dimensions, set dimension variables
    const mwSize *projDims; projDims = mxGetDimensions(prhs[2]); // dimensions of ptsArray
    unsigned int width = projDims[0]; unsigned int height = projDims[1];
    const mwSize *recdims; recdims = mxGetDimensions(prhs[0]); //dimensions of image
    unsigned int nSlices = recdims[2]; unsigned int sliceSideLength = recdims[0];
   
    //declare additional variables needed
    char s[200]; //for error messages
    //voxel size and corner of grid
    float voxelSize, bx, by, bz; 
    bx = -(float) sliceSideLength/2.0;
    by=bx;
    bz = -(float) nSlices/2.0;
    voxelSize = 1.0;
    //block/grid dimensions for CUDA execution
    dim3 dimBlock(32,32);
    dim3 dimGrid((int)ceil(width/32.f), (int) ceil(height/32.f));
    float beta, sinBeta, cosBeta;//projection angle sin and cos
    size_t projOffset = width*height; //indexing offset for projection data

    // projection volume 
    float *P;
    const mwSize dims[]={width, height, nProj};
    plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    P = (float*)mxGetData(plhs[0]);

    //allocate projection array in device memory
    float* projection;
    size_t projectionMemSize = width*height*sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&projection,projectionMemSize));
    checkCudaError("projection malloc");
    CHECK_CUDA_ERROR(cudaMemset(projection,0,projectionMemSize));

    //allocate projection points array in device memory
    float* d_ptsArray;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ptsArray, projectionMemSize*6));
    checkCudaError("d_ptsArray malloc");
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptsArray, h_ptsArray, projectionMemSize*6, cudaMemcpyHostToDevice));

    
    //allocate texture memory (3D) for image volume
    //size_t sizeTex = sliceSideLength*sliceSideLength*nSlices*sizeof(float);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent((size_t)sliceSideLength,(size_t)sliceSideLength,(size_t)nSlices);
    cudaArray *cuArray;
    //mexPrintf("\nsizeTex = %d",sizeTex);

    //cudaMallocArray(&cuArray, &channelDesc, width, width, height);
    CHECK_CUDA_ERROR(cudaMalloc3DArray(&cuArray, &channelDesc, extent, 0));
    checkCudaError("texture3D malloc");
    CHECK_CUDA_ERROR(cudaBindTextureToArray(reconTex, cuArray, channelDesc));
    checkCudaError("texture bind");
    reconTex.filterMode = cudaFilterModeLinear;
    reconTex.normalized = false;
    reconTex.addressMode[0] = cudaAddressModeBorder;
    reconTex.addressMode[1] = cudaAddressModeBorder;
    reconTex.addressMode[2] = cudaAddressModeBorder;

    //copy recon to cuda array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*) recon, sliceSideLength * sizeof(float), nSlices, sliceSideLength);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;

    CHECK_CUDA_ERROR(cudaMemcpy3D(&copyParams));

    //loop through all projection angles
    for( size_t projInd = 0; projInd < nProj; projInd++)
    {
        beta = projAngs[projInd];
        sincosf(beta, &sinBeta, &cosBeta);
        forwardProjKernel<<<dimGrid,dimBlock>>>( projection, d_ptsArray, bx, by, bz, sinBeta, cosBeta, voxelSize, sliceSideLength, sliceSideLength, nSlices, width, height);

        cudaThreadSynchronize();
        checkCudaError("main kernel invocation");

        //copy data to projection array
        CHECK_CUDA_ERROR(cudaMemcpy(&P[projInd * projOffset], projection, projectionMemSize, cudaMemcpyDeviceToHost));
    }
    
    cudaFree(projection);
    cudaUnbindTexture(reconTex);
    cudaFreeArray(cuArray);
    cudaFree(d_ptsArray);
    //cudaDeviceReset(); //may 9 2016
    return;
}

void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err )
    {
        mexErrMsgTxt(msg);
    }
}




    
    
