/*CUDAmex_forwardProjection.cu

Computes a forward projection operation for a parallel-beam, stacked fan-beam, or cone-beam CT geometry. 


Kurtis H Dekker, PhD
Department of Medical Physics,
Cancer Centre of Southeastern Ontario,
Kingston General Hospital,
Kingston, ON, CANADA

Created : July 2 2015
Modified: March 21 2016 - support array of projection angles (non-equal spacing)
		  July 23 2019  - cleanup and commenting for public release
*/

#include "mex.h"
 
// This define command is added for the M_PI constant
#define _USE_MATH_DEFINES 1
#include <math.h>
#include <cuda.h>

// macro to handle CUDA errors. Wrap CUDA mallocs and memCpy in here
#define CHECK_CUDA_ERROR(x) do {\
	cudaError_t res = (x); \
	if (res != cudaSuccess) { \
    sprintf(s,"CUDA ERROR: %s = %d (%s) at (%s:%d)\n",#x, res, cudaGetErrorString(res),__FILE__,__LINE__);\
	mexErrMsgTxt(s);\
	\
	} \
} while(0)

void checkCudaError(const char *msg);

texture<float, 3, cudaReadModeElementType> reconTex; //for storing a volume

__global__ void forwardProjKernel_par(float* projectionOut, float bx, float by, float bz, float u0, float v0, float sinBeta, float cosBeta, float SR, float RD, float voxelSize,unsigned int imSizeX, unsigned int imSizeY, unsigned int imSizeZ, unsigned int width, unsigned int height)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) return; //don't do anything if we're outside the projection

    //calculate ray position on detector (u,v)
    float u = ((float) ix - u0) ;//* (SR + RD)/SR;
    float v = ((float) iy - v0) ;//* (SR+RD)/SR;

    //calculate ray start/end points in 3d space xyz, origin at centre of recon volume

    float3 startpoint, endpoint, currentpoint;

    startpoint.x = (SR * cosBeta) - u*sinBeta;
    startpoint.y = (SR * sinBeta) + u*cosBeta;
    startpoint.z = v;
       float tval = startpoint.x; 
    endpoint.x = -RD*cosBeta - u*sinBeta;
    endpoint.y = -RD*sinBeta + u*cosBeta;
    endpoint.z = v;
    

    float dConv = sqrtf((endpoint.x - startpoint.x)*(endpoint.x-startpoint.x) + (endpoint.y - startpoint.y)*(endpoint.y-startpoint.y) + (endpoint.z - startpoint.z)*(endpoint.z-startpoint.z));

    //calculate alphaOne values (x,y,z)
    float alphaXone, alphaXtwo, alphaYone, alphaYtwo, alphaZone, alphaZtwo;

    if (fabsf(endpoint.x - startpoint.x) >= 1e-5) {
        alphaXone = (bx - startpoint.x) / (endpoint.x - startpoint.x);
        alphaXtwo = (-bx - startpoint.x) / (endpoint.x - startpoint.x);
    } else {
        alphaXone = 0.f/0.f;
        alphaXtwo = 0.f/0.f;
    }

    if (fabsf(endpoint.y - startpoint.y) >= 1e-5) {
        alphaYone = (by - startpoint.y) / (endpoint.y - startpoint.y);
        alphaYtwo = (-by - startpoint.y) / (endpoint.y - startpoint.y);
    } else {
        alphaYone = 0.f/0.f;
        alphaYtwo = 0.f/0.f;
    }

    if (fabsf(endpoint.z - startpoint.z) >= 1e-5) {
        alphaZone = (bz - startpoint.z) / (endpoint.z - startpoint.z);
        alphaZtwo = (-bz - startpoint.z) / (endpoint.z - startpoint.z);
    } else {
        alphaZone = 0.f/0.f;
        alphaZtwo = 0.f/0.f;
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
       // dval=tval;
        currentpoint.x += stepsizeX;
        currentpoint.y += stepsizeY;
        currentpoint.z += stepsizeZ;
        travelled += stepsize;
    }

    projectionOut[iy * width + ix] = dval;
    
}

__global__ void forwardProjKernel_cone(float* projectionOut, float bx, float by, float bz, float u0, float v0, float sinBeta, float cosBeta, float SR, float RD, float voxelSize,unsigned int imSizeX, unsigned int imSizeY, unsigned int imSizeZ, unsigned int width, unsigned int height)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) return; //don't do anything if we're outside the projection

    //calculate ray position on detector (u,v)
    float u = ((float) ix - u0) * (SR + RD)/SR;
    float v = ((float) iy - v0) * (SR+RD)/SR;

    //calculate ray start/end points in 3d space xyz, origin at centre of recon volume

    float3 startpoint, endpoint, currentpoint;

    startpoint.x = (SR * cosBeta );
    startpoint.y = ( SR * sinBeta );
    startpoint.z = 0.0f;
    
    endpoint.x = -RD*cosBeta - u*sinBeta;
    endpoint.y = -RD*sinBeta + u*cosBeta;
    endpoint.z = v;
    

    float dConv = sqrtf((endpoint.x - startpoint.x)*(endpoint.x-startpoint.x) + (endpoint.y - startpoint.y)*(endpoint.y-startpoint.y) + (endpoint.z - startpoint.z)*(endpoint.z-startpoint.z));

    //calculate alphaOne values (x,y,z)
    float alphaXone, alphaXtwo, alphaYone, alphaYtwo, alphaZone, alphaZtwo;

    if (fabsf(endpoint.x - startpoint.x) >= 1e-5) {
        alphaXone = (bx - startpoint.x) / (endpoint.x - startpoint.x);
        alphaXtwo = (-bx - startpoint.x) / (endpoint.x - startpoint.x);
    } else {
        alphaXone = 0.f/0.f;
        alphaXtwo = 0.f/0.f;
    }

    if (fabsf(endpoint.y - startpoint.y) >= 1e-5) {
        alphaYone = (by - startpoint.y) / (endpoint.y - startpoint.y);
        alphaYtwo = (-by - startpoint.y) / (endpoint.y - startpoint.y);
    } else {
        alphaYone = 0.f/0.f;
        alphaYtwo = 0.f/0.f;
    }

    if (fabsf(endpoint.z - startpoint.z) >= 1e-5) {
        alphaZone = (bz - startpoint.z) / (endpoint.z - startpoint.z);
        alphaZtwo = (-bz - startpoint.z) / (endpoint.z - startpoint.z);
    } else {
        alphaZone = 0.f/0.f;
        alphaZtwo = 0.f/0.f;
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
    
    float stepsize = voxelSize / 1.0f;

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

__global__ void forwardProjKernel_fan(float* projectionOut, float bx, float by, float bz, float u0, float v0, float sinBeta, float cosBeta, float SR, float RD, float voxelSize,unsigned int imSizeX, unsigned int imSizeY, unsigned int imSizeZ, unsigned int width, unsigned int height)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) return; //don't do anything if we're outside the projection

    //calculate ray position on detector (u,v)
    float u = ((float) ix - u0) * (SR + RD)/SR;
    float v = ((float) iy - v0) ;//* (SR+RD)/SR;

    //calculate ray start/end points in 3d space xyz, origin at centre of recon volume

    float3 startpoint, endpoint, currentpoint;

    startpoint.x = (SR * cosBeta );
    startpoint.y = ( SR * sinBeta );
    startpoint.z = v;
    
    endpoint.x = -RD*cosBeta - u*sinBeta;
    endpoint.y = -RD*sinBeta + u*cosBeta;
    endpoint.z = v;
    

    float dConv = sqrtf((endpoint.x - startpoint.x)*(endpoint.x-startpoint.x) + (endpoint.y - startpoint.y)*(endpoint.y-startpoint.y) + (endpoint.z - startpoint.z)*(endpoint.z-startpoint.z));

    //calculate alphaOne values (x,y,z)
    float alphaXone, alphaXtwo, alphaYone, alphaYtwo, alphaZone, alphaZtwo;

    if (fabsf(endpoint.x - startpoint.x) >= 1e-5) {
        alphaXone = (bx - startpoint.x) / (endpoint.x - startpoint.x);
        alphaXtwo = (-bx - startpoint.x) / (endpoint.x - startpoint.x);
    } else {
        alphaXone = 0.f/0.f;
        alphaXtwo = 0.f/0.f;
    }

    if (fabsf(endpoint.y - startpoint.y) >= 1e-5) {
        alphaYone = (by - startpoint.y) / (endpoint.y - startpoint.y);
        alphaYtwo = (-by - startpoint.y) / (endpoint.y - startpoint.y);
    } else {
        alphaYone = 0.f/0.f;
        alphaYtwo = 0.f/0.f;
    }

    if (fabsf(endpoint.z - startpoint.z) >= 1e-5) {
        alphaZone = (bz - startpoint.z) / (endpoint.z - startpoint.z);
        alphaZtwo = (-bz - startpoint.z) / (endpoint.z - startpoint.z);
    } else {
        alphaZone = 0.f/0.f;
        alphaZtwo = 0.f/0.f;
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
//CUDAmex_forwardProjection(image, D, projSpacing, nProj, geomFlag) -> geomFlag = 0 (par3D), 1(fan3D), 2(cone3d)
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("mex_conebeam_FP:invalidInput","4 arguments expected");

    //image volume
    float *recon;
    recon = (float *)mxGetData(prhs[0]);
    float *projAngs;
    projAngs = (float *)mxGetData(prhs[2]);
    const mwSize *dim_array;
    dim_array = mxGetDimensions(prhs[0]);
    

    float D = mxGetScalar(prhs[1]);
   // float projSpacing = mxGetScalar(prhs[2]);
    
    int nProj = mxGetScalar(prhs[3]);

    int geomFlag = mxGetScalar(prhs[4]);
    //mexPrintf("%d",geomFlag);
    // projection volume 
    float *P;
    const mwSize dims[]={dim_array[0],dim_array[2],nProj};
    plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    P = (float*)mxGetData(plhs[0]);

    //centre of detector plane
    float u0 = float(dim_array[0]-1)/2;
    float v0 = float(dim_array[2]-1)/2;

    char s [200];
    unsigned int width = dim_array[0];
    unsigned int height = dim_array[2];


    //allocate projection array in device memory
    float* projection;
    size_t projectionMemSize = width*height*sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&projection,projectionMemSize));
    checkCudaError("projection malloc");
    CHECK_CUDA_ERROR(cudaMemset(projection,0,projectionMemSize));

    //allocate texture memory (3D) for image volume
    size_t sizeTex = width*width*height*sizeof(float);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent((size_t)width,(size_t)width,(size_t)height);
    cudaArray *cuArray;
    //mexPrintf("\nsizeTex = %d",sizeTex);

    //cudaMallocArray(&cuArray, &channelDesc, width, width, height);
    CHECK_CUDA_ERROR(cudaMalloc3DArray(&cuArray, &channelDesc, extent, 0)); 
    checkCudaError("texture3D malloc");
    CHECK_CUDA_ERROR(cudaBindTextureToArray(reconTex, cuArray, channelDesc));
    checkCudaError("texture bind");
    reconTex.filterMode = cudaFilterModeLinear;
    reconTex.normalized = false;
    reconTex.addressMode[0] = cudaAddressModeClamp;
    reconTex.addressMode[1] = cudaAddressModeClamp;
    reconTex.addressMode[2] = cudaAddressModeClamp;

    //copy recon to cuda array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*) recon, width * sizeof(float), height, width);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;

    CHECK_CUDA_ERROR(cudaMemcpy3D(&copyParams));


    float voxelSize, bx, by, bz;
    int imSizeX, imSizeY, imSizeZ;

    bx = - (float) width / 2.0;
    by = - (float) width / 2.0;
    bz = - (float) height / 2.0;
    voxelSize = 1.0;
    imSizeX = width;
    imSizeY = width;
    imSizeZ = height;

    dim3 dimBlock(32,32);
    int nblocks;
   if ( width % 32 == 0){
       nblocks = width/32;
   } else {
       nblocks = width/32 + 1;
   }
   //mexPrintf("nblocks = %d",nblocks);
   dim3 dimGrid(nblocks,nblocks);

    float beta, sinBeta, cosBeta;
    size_t projOffset = width*height;

    for( size_t projInd = 0; projInd < nProj; projInd++)
    {
        beta = projAngs[projInd];
      
        sincosf(beta, &sinBeta, &cosBeta);

        switch (geomFlag)
        {
            case 0:
                forwardProjKernel_par<<<dimGrid,dimBlock>>>(projection, bx, by, bz, u0, v0, sinBeta, cosBeta, D, D, voxelSize, imSizeX, imSizeY, imSizeZ, width, height);
                break;
            case 1:
                forwardProjKernel_fan<<<dimGrid,dimBlock>>>(projection, bx, by, bz, u0, v0, sinBeta, cosBeta, D, D, voxelSize, imSizeX, imSizeY, imSizeZ, width, height); 
                break;
            case 2:
                forwardProjKernel_cone<<<dimGrid,dimBlock>>>(projection, bx, by, bz, u0, v0, sinBeta, cosBeta, D, D, voxelSize, imSizeX, imSizeY, imSizeZ, width, height); 
                break;
            default:
                mexErrMsgTxt("invalid geometry");
        }


        cudaThreadSynchronize();
        checkCudaError("main kernel invocation");

        //copy data to projection array
        CHECK_CUDA_ERROR(cudaMemcpy(&P[projInd * projOffset], projection, projectionMemSize, cudaMemcpyDeviceToHost));
    }
    cudaFree(projection);
    cudaUnbindTexture(reconTex);
    cudaFreeArray(cuArray);
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




    
    
