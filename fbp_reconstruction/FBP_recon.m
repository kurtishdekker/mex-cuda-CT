function recon = FBP_recon(pre,post, scanAngles, geom,filter,filterCutoff)
%FBP_RECON.M - gpu reconstruction with FBP
%
%Inputs:
%       pre - reference scan (MxNxNproj, eg. 480x640x1024)
%       post - data scan
%       geom - the structure specifying fan, cone, or
%       scanAngles - the projection angles
%       filter - the filter type, see preFilterSinogram.m for options
%       filterCutoff - the normalized filter cutoff frequency
%
%Output:
%       recon - final reconstruction dataset. Size will be determined by
%       projection size
%
%Dependencies:
%       NVIDIA GPU with sufficient memory for recon grid
%       CUDA toolbox v6.0 or higher
%
%Created:  April 10 2017 by Kurtis H. Dekker
%Modified: July 25 2019 by KHD
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% perfom sinogram log transform and filtering
sino = -log(post./pre);
sino(isnan(sino)) = 0;
sino(isinf(sino)) = 0;

filteredSino = single(preFilterSinogram(sino,geom,scanAngles,filter,filterCutoff))
%% backproject

if strcmpi(geom.type,'par3d')
    recon = CUDAmex_BP(filteredSino,geom.SAD,scanAngles,0);
elseif strcmpi(geom.type,'fan3d')
    recon = CUDAmex_BP(filteredSino,geom.SAD,scanAngles,1);
elseif strcmpi(geom.type,'cone')
    recon = CUDAmex_BP(filteredSino,geom.SAD,scanAngles,2);
else
	error('Invalid Geometry Selection');
end