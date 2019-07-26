/*CUDAmex_OSCiter_par3D.cu

Mex source code.
Compute 1 iteration of the OSC algorithm using the GPU.

Usage (Matlab):
	output = CUDAmex_OSCiter_par3D(reconVol, preScan, postScan, numSubsets, scanAngles);

Inputs:
	reconVol 		-		the current estimate of the 3D image 
	preScan	 		-		pre-scan projection data 
	postScan		-		post-scan projection data
	numSubsets		-		number of subsets to divide projections into on the current iteration
    subsetPicks     -       order of subsets
	scanAngles		-		vector of angles (in radians) at which projections are acquired

Outputs:
	output 			-		the updated 3D image volume for the iteration
	
Dependencies:
	CUDA toolkit v6.0 or later
	NVIDIA GPU with compute capability 3.5 or higher and enough vRAM to store 3 arrays of size Nx x Ny x Nz (recon size)
	
NOTES:
	uses single precision. input arrays in matlab should be cast as type single

TODO:
	try to optimize to reduce amount of gpu vRAM required if necessary, and/or implement code to perform sub-volume reconstructions on larger images
	
	

Author   : Kurtis H Dekker, PhD
Created  : April 10 2017
Modified : July 23, 2019

*/

//INCLUDES AND DEFINES
#define _USE_MATH_DEFINES 1

#include "mex.h"

#include <math.h>
#include <cuda.h>
#include <math_constants.h> //for CUDART_NAN_F

//thrust library
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


//FUNCTION DECLARATIONS
void checkCudaError(const char *msg);
__global__ void oscSetCorrTermZero(unsigned int width, unsigned int height);
__global__ void oscIterUpdateKernel(unsigned int width, unsigned int height);
__global__ void oscIterForwardProjKernel(float bx, float by, float bz, float u0, float v0, float sinBeta, float cosBeta, float SR, float RD, float voxelSize, unsigned int imSizeX, unsigned int imSizeY, unsigned int imSizeZ, unsigned int width, unsigned int height, int geomFlag);
__global__ void oscIterBackProjKernel( float v0, float sinBeta, float cosBeta, float SR, unsigned int width, unsigned int height, int geomFlag);


//DECLARE SURFACE AND TEXTURES
texture<float, 2, cudaReadModeElementType> preScanProjTex; //for storing a projection
texture<float, 2, cudaReadModeElementType> postScanProjTex; //for storing projection
texture<float, 3, cudaReadModeElementType> reconTex; //for storing a volume
surface<void, 3> reconSurf; //for storing volume 
surface<void, 3> correctionVolumeNumeratorSurf; // correction term for volume update
surface<void, 3> correctionVolumeDenominatorSurf; // correction term 2 for update
surface<void, 2> preSurf; //preScan projection
surface<void, 2> postSurf; //postScan projection


__global__ void oscSetCorrTermZero(unsigned int width, unsigned int height)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    

    unsigned int halfwidth = width/2 - 1;
    if( ((ix-halfwidth)*(ix-halfwidth) + (iy-halfwidth)*(iy-halfwidth)) < halfwidth*halfwidth)
{
    
    for( int iz = 0; iz < height; iz++)
    {

        surf3Dwrite(0.f , correctionVolumeNumeratorSurf, ix *sizeof(float), iy, iz); //3D surface write
        surf3Dwrite(0.f , correctionVolumeDenominatorSurf, ix *sizeof(float), iy, iz); //3D surface write
        
    }
}
}
__global__ void oscIterUpdateKernel(unsigned int width, unsigned int height)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    

    unsigned int halfwidth = width/2 - 1;
    if( ((ix-halfwidth)*(ix-halfwidth) + (iy-halfwidth)*(iy-halfwidth)) < halfwidth*halfwidth)
    {

    float cval, cval2, fval;
    
    for( int iz = 0; iz < height; iz++)
    {
        surf3Dread(&cval, correctionVolumeNumeratorSurf, ix * sizeof(float),iy,iz);
        surf3Dread(&cval2, correctionVolumeDenominatorSurf, ix * sizeof(float),iy,iz);
        surf3Dread(&fval, reconSurf, ix * sizeof(float), iy, iz);
        //cval = tex3D(reconTex,  
        surf3Dwrite(fmaxf(fval+(fval*cval/cval2),0.f) , reconSurf, ix *sizeof(float), iy, iz); //3D surface write
        surf3Dwrite(0.f, correctionVolumeNumeratorSurf, ix * sizeof(float), iy, iz);
        surf3Dwrite(0.f, correctionVolumeDenominatorSurf, ix * sizeof(float), iy, iz);
    }
}
}



__global__ void oscIterForwardProjKernel(float bx, float by, float bz, float u0, float v0, float sinBeta, float cosBeta, float SR, float RD, float voxelSize, unsigned int imSizeX, unsigned int imSizeY, unsigned int imSizeZ, unsigned int width, unsigned int height, int geomFlag)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) return; //don't do anything if we're outside the projection
    //calculate ray position on detector (u,v)
    float u, v;

    //calculate ray start/end points in 3d space xyz, origin at centre of recon volume

    float3 startpoint, endpoint, currentpoint;
    switch (geomFlag)
    {
        case 0: //par
            u = ((float) ix - u0) ;
            v = ((float) iy - v0) ;
            startpoint.x = (SR * cosBeta) - u*sinBeta;
            startpoint.y = (SR * sinBeta) + u*cosBeta;
            startpoint.z = v;
            //float tval = startpoint.x; 
            endpoint.x = -RD*cosBeta - u*sinBeta;
            endpoint.y = -RD*sinBeta + u*cosBeta;
            endpoint.z = v;
            break;
        case 1: //fan
            u = ((float) ix - u0) * (SR + RD)/SR;
            v = ((float) iy - v0) ;
            startpoint.x = (SR * cosBeta) ;
            startpoint.y = (SR * sinBeta) ;
            startpoint.z = v;
            //float tval = startpoint.x; 
            endpoint.x = -RD*cosBeta - u*sinBeta;
            endpoint.y = -RD*sinBeta + u*cosBeta;
            endpoint.z = v;     
            break;
        case 2: //cone
            u = ((float) ix - u0) * (SR + RD)/SR;
            v = ((float) iy - v0) * (SR+RD)/SR; 
            startpoint.x = (SR * cosBeta);
            startpoint.y = (SR * sinBeta);
            startpoint.z = 0.f;
            //float tval = startpoint.x; 
            endpoint.x = -RD*cosBeta - u*sinBeta;
            endpoint.y = -RD*sinBeta + u*cosBeta;
            endpoint.z = v;
        break;
    }
        
    

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
       // dval=tval;
        currentpoint.x += stepsizeX;
        currentpoint.y += stepsizeY;
        currentpoint.z += stepsizeZ;
        travelled += stepsize;
    }
	

    float preval, postval;
    surf2Dread(&preval, preSurf, ix*sizeof(float), iy);
    surf2Dread(&postval, postSurf, ix*sizeof(float), iy);
    
    surf2Dwrite((preval * expf(-dval) - postval) , preSurf,ix*sizeof(float), iy);
    // surf2Dwrite(dval,preSurf,ix*4,iy);
    surf2Dwrite(preval * expf(-dval)*dval, postSurf, ix*sizeof(float), iy);
	
    
}
__global__ void oscIterBackProjKernel( float v0, float sinBeta, float cosBeta, float SR, unsigned int width, unsigned int height, int geomFlag)
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
        float s = x*cosBeta + y*sinBeta;
        float t = -x*sinBeta + y*cosBeta;
        float U, UU;
        float proj_y, proj_z;
        switch (geomFlag)
        {
            case 0:
                
                U = 1.f;
                UU = 1.f;
                proj_y= t*UU + halfwidth + 1.f;
                proj_z= -v0*UU + v0 + 0.5f;
                break;
            case 1: //fan
                U = 1.f;
                UU = SR / (SR-s);
                proj_y = t*UU + halfwidth + 1.f;
                proj_z = 0.5f;
                break;
            case 2: //cone
                U = SR / (SR - s);
                UU = SR/ (SR-s);
                proj_y = t*U + halfwidth + 1.f;
                proj_z = -v0*UU + v0 + 0.5f;
                break;
        }



        //loop over 'z' in image space (vertical direction) and backproject pixel value
        for ( unsigned int iz = 0; iz < height; iz++)
        {
            //output[ix + width*iy + width*width*iz] += U*U * tex2D(preScanProjTex,proj_y,proj_z);
            float fval = UU*UU * tex2D(preScanProjTex, proj_y, proj_z);
            float cval;
			
            surf3Dread(&cval, correctionVolumeNumeratorSurf, ix * 4,iy,iz);
            surf3Dwrite(cval+(fval) , correctionVolumeNumeratorSurf, ix *sizeof(float), iy, iz); //3D surface write
            
            fval = UU*UU*tex2D(postScanProjTex, proj_y, proj_z);
            surf3Dread(&cval, correctionVolumeDenominatorSurf, ix * 4,iy,iz);
            surf3Dwrite(cval+(fval) , correctionVolumeDenominatorSurf, ix *sizeof(float), iy, iz); //3D surface write
            proj_z += U;

        }
    }
}
// the mex function wrapper
//compute 1 iteration of the OSC part of OSC-TV
//CUDAmex_OSCiter_par3D(recon, pre,post,numSubsets,angles) 
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("mex_conebeam_FP:invalidInput","4 arguments expected");

    //image volume
    float *recon;
    recon = (float *)mxGetData(prhs[0]);
    float *projAngs;
    projAngs = (float *)mxGetData(prhs[5]);
    const mwSize *dim_array;
    dim_array = mxGetDimensions(prhs[0]);

    float D = mxGetScalar(prhs[6]);
    int geomFlag = mxGetScalar(prhs[7]);
    unsigned int width = dim_array[0];
    unsigned int height = dim_array[2];
	
	float *pre_data;
	pre_data = (float *)mxGetData(prhs[1]);
	float *post_data;
	post_data = (float *)mxGetData(prhs[2]);
    
	const mwSize *projDims;
	projDims = mxGetDimensions(prhs[1]);
	int nProj = projDims[2];
    int numSubsets = mxGetScalar(prhs[3]);
    int *subsetPicks;
    subsetPicks = (int *)mxGetData(prhs[4]);

    // output volume
    //float *output;
    //size_t outputMemSize = width * width * sizeof(float)*(height);
    
    //const mwSize dims[]={dim_array[0],dim_array[1],dim_array[2]};
    //plhs[0] = mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    //output = (float*)mxGetData(plhs[0]);

    //centre of detector plane
    float u0 = float(dim_array[0]-1)/2;
    float v0 = float(dim_array[2]-1)/2;

    char s [200];
    //mexPrintf("\nheight = %d, width = %d",height,width);
    //mexPrintf("\nu0 = %g, v0 = %g",u0,v0);
    //mexPrintf("\nD = %g, projSpacing = %g",D,projSpacing);


	// allocate texture memory on GPU for projections
   size_t sizeProjTex = width*height*sizeof(float);
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
   cudaArray *preProjArray_d;
   cudaArray *postProjArray_d;


    preScanProjTex.filterMode = cudaFilterModeLinear;
    preScanProjTex.normalized = false;
    preScanProjTex.addressMode[0] = cudaAddressModeBorder;
    preScanProjTex.addressMode[1] = cudaAddressModeBorder;
	
	postScanProjTex.filterMode = cudaFilterModeLinear;
    postScanProjTex.normalized = false;
    postScanProjTex.addressMode[0] = cudaAddressModeBorder;
    postScanProjTex.addressMode[1] = cudaAddressModeBorder;

    //malloc proj arrays
    CHECK_CUDA_ERROR(cudaMallocArray(&preProjArray_d, &channelDesc, width, height, cudaArraySurfaceLoadStore));
    CHECK_CUDA_ERROR(cudaMallocArray(&postProjArray_d, &channelDesc, width, height, cudaArraySurfaceLoadStore));

    //bind tex2D to prescan proj array
    CHECK_CUDA_ERROR(cudaBindTextureToArray(preScanProjTex, preProjArray_d, channelDesc));
    CHECK_CUDA_ERROR(cudaBindTextureToArray(postScanProjTex, postProjArray_d, channelDesc));


    //bind surfaces to proj arrays
    CHECK_CUDA_ERROR(cudaBindSurfaceToArray(preSurf, preProjArray_d ));
    CHECK_CUDA_ERROR(cudaBindSurfaceToArray(postSurf, postProjArray_d ));
   
    //allocate texture memory (3D) for image volume
    //size_t sizeTex = width*width*height*sizeof(float);
   // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent((size_t)width,(size_t)width,(size_t)height);
    cudaArray *reconVolumeArray_d;
    //mexPrintf("\nsizeTex = %d",sizeTex);
    cudaArray *correctionVolumeNumeratorArray_d;
    cudaArray *correctionVolumeDenominatorArray_d;
    	

    //cudaMallocArray(&reconVolumeArray_d, &channelDesc, width, width, height);
    CHECK_CUDA_ERROR(cudaMalloc3DArray(&reconVolumeArray_d, &channelDesc, extent, cudaArraySurfaceLoadStore)); 
    CHECK_CUDA_ERROR(cudaMalloc3DArray(&correctionVolumeNumeratorArray_d, &channelDesc, extent, cudaArraySurfaceLoadStore)); 
    CHECK_CUDA_ERROR(cudaMalloc3DArray(&correctionVolumeDenominatorArray_d, &channelDesc, extent, cudaArraySurfaceLoadStore)); 
    
    
    CHECK_CUDA_ERROR(cudaBindTextureToArray(reconTex, reconVolumeArray_d, channelDesc));

    reconTex.filterMode = cudaFilterModeLinear;
    reconTex.normalized = false;
    reconTex.addressMode[0] = cudaAddressModeBorder;
    reconTex.addressMode[1] = cudaAddressModeBorder;
    reconTex.addressMode[2] = cudaAddressModeBorder;

	//also bind surface to the same array for writing
	CHECK_CUDA_ERROR(cudaBindSurfaceToArray(reconSurf, reconVolumeArray_d, channelDesc));
	CHECK_CUDA_ERROR(cudaBindSurfaceToArray(correctionVolumeNumeratorSurf, correctionVolumeNumeratorArray_d, channelDesc));
	CHECK_CUDA_ERROR(cudaBindSurfaceToArray(correctionVolumeDenominatorSurf, correctionVolumeDenominatorArray_d, channelDesc));


	
    //copy recon to cuda array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*) recon, width * sizeof(float), height, width);
    copyParams.dstArray = reconVolumeArray_d;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;

    CHECK_CUDA_ERROR(cudaMemcpy3D(&copyParams));


    float bx, by, bz;
    int imSizeX, imSizeY, imSizeZ;

    bx = - (float) width / 2.0;
    by = - (float) width / 2.0;
    bz = - (float) height / 2.0;
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

    oscSetCorrTermZero<<<dimGrid,dimBlock>>>(width,height);		
	
	for( size_t subInd = 0; subInd < numSubsets; subInd++ )
	{
		int subsetSel = subsetPicks[subInd]; //todo: randomize and use matenine's method for selecting subset indices (can do in ML and pass to mex)
	    //CHECK_CUDA_ERROR(cudaMemset(correctionVolumeNumeratorArray_d,0,outputMemSize));	
		for( size_t projInd = subsetSel; projInd < nProj; projInd+=numSubsets )
		{
            
		     
			//calculate forward projection and correction term to backproject
			beta = projAngs[projInd];
		    sincosf(beta, &sinBeta, &cosBeta);
			CHECK_CUDA_ERROR(cudaMemcpyToArray(preProjArray_d, 0, 0, &pre_data[ projInd * projOffset ], sizeProjTex, cudaMemcpyHostToDevice));
			CHECK_CUDA_ERROR(cudaMemcpyToArray(postProjArray_d, 0, 0, &post_data[ projInd * projOffset ], sizeProjTex, cudaMemcpyHostToDevice));
        

			
			//compute y and p and store to pre and post proj arrays on device, since we won't access those again until the next iteration
			oscIterForwardProjKernel<<<dimGrid,dimBlock>>>(bx, by, bz, u0, v0, sinBeta, cosBeta, D, D, 1.0, imSizeX, imSizeY, imSizeZ, width, height,geomFlag);
			cudaThreadSynchronize();
			checkCudaError("main kernel invocation 1 \n");

            CHECK_CUDA_ERROR(cudaMemcpyFromArray(&pre_data[ projInd * projOffset ], preProjArray_d, 0, 0, sizeProjTex, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpyFromArray(&post_data[ projInd * projOffset ], postProjArray_d, 0, 0, sizeProjTex, cudaMemcpyDeviceToHost));
            
        }

        for( size_t projInd = subsetSel; projInd < nProj; projInd+=numSubsets ) 
        {
            //calculate forward projection and correction term to backproject
			beta = projAngs[projInd];
		    sincosf(beta, &sinBeta, &cosBeta);
			CHECK_CUDA_ERROR(cudaMemcpyToArray(preProjArray_d, 0, 0, &pre_data[ projInd * projOffset ], sizeProjTex, cudaMemcpyHostToDevice));
			CHECK_CUDA_ERROR(cudaMemcpyToArray(postProjArray_d, 0, 0, &post_data[ projInd * projOffset ], sizeProjTex, cudaMemcpyHostToDevice));
            
			//backproject correction term and update on device
			oscIterBackProjKernel<<<dimGrid,dimBlock>>> ( v0, sinBeta, cosBeta, D, width, height, geomFlag);
            //mexPrintf("reached post-BP stage %d \n",projInd);
			cudaThreadSynchronize();
			checkCudaError("main kernel invocation 2 \n");


		}
        //update image estimate
        oscIterUpdateKernel<<<dimGrid,dimBlock>>> ( width, height );
		
			
			
			
		cudaThreadSynchronize();
		checkCudaError("main kernel invocation 3 \n");

			
	}
    //float* ptrRecon;	
	//ptrRecon = output;
	//cudaMemcpy(ptrRecon, reconVolumeArray_d, outputMemSize, cudaMemcpyDeviceToHost);
    //CHECK_CUDA_ERROR(cudaMemcpyFromArray(ptrRecon, reconVolumeArray_d, 0, 0, outputMemSize, cudaMemcpyDeviceToHost));
	
    cudaMemcpy3DParms copyParams2 = {0};
    copyParams2.srcArray = reconVolumeArray_d;
    copyParams2.dstPtr = make_cudaPitchedPtr((void*) recon, width * sizeof(float), height, width);
    copyParams2.extent = extent;
    copyParams2.kind = cudaMemcpyDeviceToHost;

    CHECK_CUDA_ERROR(cudaMemcpy3D(&copyParams2));

	//cudaFree(output);
    cudaUnbindTexture(reconTex);
	cudaUnbindTexture(preScanProjTex);
    cudaUnbindTexture(postScanProjTex);

    cudaFreeArray(reconVolumeArray_d); //recon
    cudaFreeArray(correctionVolumeNumeratorArray_d);
    cudaFreeArray(correctionVolumeDenominatorArray_d);
	cudaFreeArray(preProjArray_d); //preproj
	cudaFreeArray(postProjArray_d); //postproj
    cudaDeviceReset();
    return;
}

void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err )
    {
        char s[200];
        sprintf(s,"CUDA ERROR: %d (%s))\n", err, cudaGetErrorString(err));
    mexWarnMsgTxt(msg); 
    mexErrMsgTxt(s);
       
    }
}




    
    
