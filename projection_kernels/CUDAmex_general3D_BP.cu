/*CUDAmex_general3D_BP.cu

Computes a backprojection operation for a "general" geometry CT dataset wherein projection ray paths are specified by arrays of start and end points rather than a typical SAD/SDD specification.


Kurtis H Dekker, PhD
Department of Medical Physics,
Cancer Centre of Southeastern Ontario,
Kingston General Hospital,
Kingston, ON, CANADA

Created : July 2 2015
Modified: March 21 2016 - support array of projection angles (non-equal spacing)
		  July 23 2019  - cleanup and commenting for public release
*/


//INCLUDES AND DEFINES

#include "mex.h"
 
// This define command is added for the M_PI constant
#define _USE_MATH_DEFINES 1

#include <math.h>
#include <cuda.h>

// define a NaN value and an Inf value that cannot be compiler-optimized away
#define CUDART_NAN_F __int_as_float(0x7fffffff)
#define CUDART_INF_F __int_as_float(0x7f800000)

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
__device__ float calcLength(float alpha, float alphaCurrent, float dConv);
__device__ float phi(float alpha, float b, float d, float p1, float p2);
__device__ void update(float *alpha, int *ind, float d, float p1, float p2);



//backprojection kernel
__global__ void backProjKernel( float* recon, float* projection, float* ptsArray, float bx, float by, float bz, float sinBeta, float cosBeta, float voxelSize, unsigned int imSizeX, unsigned int imSizeY, unsigned int imSizeZ, unsigned int width, unsigned int height, unsigned int nProj )
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) return; //don't do anything if we're outside the projection

//calculate ray start/end points in 3d space xyz, origin at centre of recon volume
    //define start / end points. 
    float3 startpoint, endpoint, currentpoint;
    int npts = width*height;
  
    startpoint.x = (ptsArray[iy * width + ix])*cosBeta - (ptsArray[iy*width+ix+npts])*sinBeta;
    startpoint.y = (ptsArray[iy * width + ix])*sinBeta + (ptsArray[iy*width+ix+npts])*cosBeta;
    startpoint.z = (ptsArray[iy * width + ix + 2*npts]);

    endpoint.x = (ptsArray[iy * width + ix + 3*npts])*cosBeta - (ptsArray[iy*width+ix + 4*npts])*sinBeta;
    endpoint.y = (ptsArray[iy * width + ix + 3*npts])*sinBeta + (ptsArray[iy*width+ix + 4*npts])*cosBeta;
    endpoint.z = (ptsArray[iy * width + ix + 5*npts]);

    //calculate alphaOne values (x,y,z) (where ray enters/exits the GRID)
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

    //calculate minimum alpha values
    float alphaXmin = fminf(alphaXone,alphaXtwo);
    float alphaXmax = fmaxf(alphaXone, alphaXtwo);
    float alphaYmin = fminf(alphaYone,alphaYtwo);
    float alphaYmax = fmaxf(alphaYone, alphaYtwo);
    float alphaZmin = fminf(alphaZone,alphaZtwo);
    float alphaZmax = fmaxf(alphaZone, alphaZtwo);
    
    //selecting overall alpha min and max (compare vs. 0 / 1 respectively to handle rays originating / ending within grid)
    float tmp = fmaxf(alphaXmin,fmaxf(alphaYmin, alphaZmin));
    float alphaMin = fmaxf(0.f,tmp); 
    tmp = fminf(alphaXmax, fminf(alphaYmax, alphaZmax));
    float alphaMax = fminf(1.f,tmp);
 
    // the ray is now parameterized by alpha. alphaMin corresponds to ray entry point, alphaMax to exit point
    // compute the distance travelled by the ray overall (used for calculation in the parameterized rayline)
	float dConv = sqrtf((endpoint.x-startpoint.x)*(endpoint.x-startpoint.x) + (endpoint.y-startpoint.y)*(endpoint.y-startpoint.y) + (endpoint.z-startpoint.z)*(endpoint.z-startpoint.z));


     currentpoint.x = startpoint.x + alphaMin * (endpoint.x - startpoint.x);
    currentpoint.y = startpoint.y + alphaMin * (endpoint.y - startpoint.y);
    currentpoint.z = startpoint.z + alphaMin * (endpoint.z - startpoint.z);
    
    endpoint.x = startpoint.x + alphaMax * (endpoint.x - startpoint.x);
    endpoint.y = startpoint.y + alphaMax * (endpoint.y - startpoint.y);
    endpoint.z = startpoint.z + alphaMax * (endpoint.z - startpoint.z);
    
    float distance = sqrt((endpoint.x-currentpoint.x)*(endpoint.x-currentpoint.x) + (endpoint.y-currentpoint.y)*(endpoint.y-currentpoint.y) + (endpoint.z-currentpoint.z)*(endpoint.z-currentpoint.z));

    float travelled = 0.f;
    
    float stepsize = voxelSize /2.0f;

    float stepsizeX = (endpoint.x - currentpoint.x) / distance * stepsize;
    float stepsizeY = (endpoint.y - currentpoint.y) / distance * stepsize;
    float stepsizeZ = (endpoint.z - currentpoint.z) / distance * stepsize;

    int iU,iL, jU,jL,kU,kL;
    float interpX,interpXX,interpY,interpYY,interpZ,interpZZ;
   
    while(travelled < distance) {
        
        //determine lower / upper bounds
        iL = (int) floor(currentpoint.x - (bx));
        iU = (int) ceil(currentpoint.x - (bx ));
        jL = (int) floor(currentpoint.y - (by));
        jU = (int) ceil(currentpoint.y - (by ));
        kL = (int) floor(currentpoint.z - (bz));
        kU = (int) ceil(currentpoint.z - (bz ));
        int add = 1;
        if (iL < 0 || jL < 0 || kL < 0 || iU >= imSizeX || jU >= imSizeY || kU >= imSizeZ) add=0;

        if(add) {
        interpXX = currentpoint.x - bx - iL + 0.5f; interpX = 1-interpXX;
        interpYY = currentpoint.y - by - jL + 0.5f; interpY = 1-interpYY;
        //interpZZ = currentpoint.z - bz - kL + 0.5f; interpZ = 1-interpZZ;

        //interpX = currentpoint.x - bx - iL + 0.5f; interpXX = 1-interpX;
        //interpY = currentpoint.y - by - jL + 0.5f; interpYY = 1-interpY;
        interpZ = currentpoint.z - bz - kL + 0.5f; interpZZ = 1-interpZ;

        atomicAdd(&recon[jL * imSizeX + kL * imSizeX * imSizeY + iL],projection[iy * width + ix] * stepsize * interpXX * interpYY * interpZZ);
        atomicAdd(&recon[jL * imSizeX + kU * imSizeX * imSizeY + iL],projection[iy * width + ix] * stepsize * interpXX * interpYY * interpZ);

        /* 
        atomicAdd(&recon[jL * imSizeX + kL * imSizeX * imSizeY + iU],projection[iy * width + ix] * stepsize * interpX * interpYY * interpZZ);
        atomicAdd(&recon[jU * imSizeX + kL * imSizeX * imSizeY + iL],projection[iy * width + ix] * stepsize * interpXX * interpY * interpZZ);
        atomicAdd(&recon[jU * imSizeX + kL * imSizeX * imSizeY + iU],projection[iy * width + ix] * stepsize * interpX * interpY * interpZZ);
        atomicAdd(&recon[jU * imSizeX + kU * imSizeX * imSizeY + iL],projection[iy * width + ix] * stepsize * interpXX * interpY * interpZ);
        atomicAdd(&recon[jL * imSizeX + kU * imSizeX * imSizeY + iU],projection[iy * width + ix] * stepsize * interpX * interpYY * interpZ);
        atomicAdd(&recon[jU * imSizeX + kU * imSizeX * imSizeY + iU],projection[iy * width + ix] * stepsize * interpX * interpY * interpZ);
        */
        
        }

        currentpoint.x +=stepsizeX;
        currentpoint.y +=stepsizeY;
        currentpoint.z +=stepsizeZ;
        travelled +=stepsize;
    } 
}



// the mex function wrapper
//backprojectedIm = CUDAmex_general3D_BP(projections,projAngles,singleProjPoints,sliceSideLength,nSlices)
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("mex_conebeam_FP:invalidInput","4 arguments expected");

    
    //get mex arguments
    float *P; P = (float *)mxGetData(prhs[0]); //projection data
    float* projAngs; projAngs = (float *)mxGetData(prhs[1]); //projection angles
    float* h_ptsArray; h_ptsArray = (float *)mxGetData(prhs[2]); //points array (mxnx6)
    unsigned int sliceSideLength = mxGetScalar(prhs[3]);
    unsigned int nSlices = mxGetScalar(prhs[4]);

    //get dimensions, set dimension variables
    const mwSize *projDims; projDims = mxGetDimensions(prhs[0]); // dimensions of ptsArray
    unsigned int width = projDims[0]; unsigned int height = projDims[1]; unsigned int nProj = projDims[2];
    
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
  
    // reconstruction volume
    float *recon;
    const mwSize recdims[]={sliceSideLength,sliceSideLength,nSlices};
    plhs[0] = mxCreateNumericArray(3, recdims, mxSINGLE_CLASS, mxREAL);
    recon = (float*)mxGetData(plhs[0]);
           
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

    //allocate memory for reconstruction on device
   float* output;
   size_t outputMemSize = sliceSideLength * sliceSideLength * sizeof(float)*(nSlices);
   cudaMalloc((void**)&output, outputMemSize);
   checkCudaError("output malloc");
   cudaMemset(output,0,outputMemSize);

    
    for( size_t projInd = 0; projInd < nProj; projInd++)
    {
        beta = projAngs[projInd]; //angle of projection
        sincosf(beta, &sinBeta, &cosBeta); //sine and cosine of projAngle
        
        //copy the 'projInd'th projection to GPU
        CHECK_CUDA_ERROR(cudaMemcpy(projection, &P[ projInd*projOffset ], projectionMemSize, cudaMemcpyHostToDevice));
        
        //call the backprojection kernel
       backProjKernel<<<dimGrid,dimBlock>>>( output, projection, d_ptsArray, bx, by, bz, sinBeta, cosBeta, voxelSize, sliceSideLength, sliceSideLength, nSlices, width, height, nProj);

        //synchronize threads (unsure if required) and check error state of CUDA instance 
        cudaThreadSynchronize();
        checkCudaError("main kernel invocation");

    }
    
    // copy recon volume to host
    CHECK_CUDA_ERROR(cudaMemcpy(recon,output,outputMemSize,cudaMemcpyDeviceToHost));
    checkCudaError("output error");
    //mexPrintf("array copied from device memory\n");
    
    // free GPU resources
    cudaFree(output);
    cudaFree(projection);
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

//HELPER DEVICE FUNCTIONS
//calculate length of intersection in Jacob's algorithm
__device__ float calcLength(float alpha, float alphaCurrent, float dConv) 
{
	return fabsf((alpha - alphaCurrent) * dConv);
}

//calculate "phi" in Jacob's algorithm
__device__ float phi(float alpha, float b, float d, float p1, float p2)
{
	float p = p1 + alpha * (p2-p1);
	return ((p - b) / d);
}

//update alpha value in Jacob's algorithm
__device__ void update(float *alpha, int *ind, float d, float p1, float p2)
{
	float tmp = *alpha + d/fabsf(p2-p1);
	*alpha =tmp;
}


    
    

