% ENVIRONMENT VARIABLES AND PATHS
cuda_path = getenv('CUDA_PATH');
if ~isempty(strfind(computer('arch'),'64'))
    cuda_path = [cuda_path '\lib\x64'];
else
    cuda_path = [cuda_path '\lib\Win32'];
end

matlab_path = [matlabroot '\extern\include'];
vs_path = 'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\' %this will need to be changed for your installation of visual studio
gpu_compute = '-gencode=arch=compute_61,code=sm_61'; %change the numbers for your specific GPU compute capability (i.e. compute capability 5.2 -> 52, etc.)

user_matlab_path = [userpath '\mex-cuda-CT'];
if ~isdir(user_matlab_path)
    mkdir(user_matlab_path)
end


%% Forward and Backprojection Operators
cd .\projection_kernels

% Compile the CUDA code into object files (.o)
% forward projection operations
nvccCommandLine = ['nvcc -c CUDAmex_FP.cu -o CUDAmex_FP.o -O3 ' gpu_compute ' --ptxas-options=-v -I"' matlab_path '" -l"' vs_path '\lib" -ccbin "' vs_path '\bin"'];
status = system(nvccCommandLine);
if status < 0
	error 'Error invoking nvcc'
end

% general3D forward projection operations
nvccCommandLine = ['nvcc -c CUDAmex_general3D_FP.cu -o CUDAmex_general3D_FP.o -O3 ' gpu_compute ' --ptxas-options=-v -I"' matlab_path '" -l"' vs_path '\lib" -ccbin "' vs_path '\bin"'];
status = system(nvccCommandLine);
if status < 0
	error 'Error invoking nvcc'
end

% backprojection operations
nvccCommandLine = ['nvcc -c CUDAmex_BP.cu -o CUDAmex_BP.o -O3 ' gpu_compute ' --ptxas-options=-v -I"' matlab_path '" -l"' vs_path '\lib" -ccbin "' vs_path '\bin"'];
status = system(nvccCommandLine);
if status < 0
	error 'Error invoking nvcc'
end

% forward projection operations
nvccCommandLine = ['nvcc -c CUDAmex_general3D_BP.cu -o CUDAmex_general3D_BP.o -O3 ' gpu_compute ' --ptxas-options=-v -I"' matlab_path '" -l"' vs_path '\lib" -ccbin "' vs_path '\bin"'];
status = system(nvccCommandLine);
if status < 0
	error 'Error invoking nvcc'
end



% compile the object files into .mex files
mex('-largeArrayDims','CUDAmex_FP.o', ['-L' cuda_path],'-lcudart');	
mex('-largeArrayDims','CUDAmex_BP.o', ['-L' cuda_path],'-lcudart');	
mex('-largeArrayDims','CUDAmex_general3D_FP.o', ['-L' cuda_path],'-lcudart');	
mex('-largeArrayDims','CUDAmex_general3D_BP.o', ['-L' cuda_path],'-lcudart');	

movefile('*.mex*', user_matlab_path);

% clean up .o files
delete *.o

%% OSC-TV iterative Reconstruction
cd ..\iterative_reconstruction
% forward projection operations
nvccCommandLine = ['nvcc -c CUDAmex_oscIter.cu -o CUDAmex_oscIter.o -O3 ' gpu_compute ' --ptxas-options=-v -I"' matlab_path '" -l"' vs_path '\lib" -ccbin "' vs_path '\bin"'];
status = system(nvccCommandLine);
if status < 0
	error 'Error invoking nvcc'
end

% TV-minimization
nvccCommandLine = ['nvcc -c CUDAmex_TVmin_3D.cu -o CUDAmex_TVmin_3D.o -O3 ' gpu_compute ' --ptxas-options=-v -I"' matlab_path '" -l"' vs_path '\lib" -ccbin "' vs_path '\bin"'];
status = system(nvccCommandLine);
if status < 0
	error 'Error invoking nvcc'
end

% compile the object files into .mex files
mex('/NODEFAULTLIB:MSVCRT.lib','-largeArrayDims', 'CUDAmex_oscIter.o', ['-L' cuda_path], '-lcudart');
mex('/NODEFAULTLIB:MSVCRT.lib','-largeArrayDims', 'CUDAmex_oscIter.o', ['-L' cuda_path], '-lcudart');

movefile('*.mex*', user_matlab_path);

% clean up .o files
delete *.o

%% add compiled mex functions to MATLAB path
addpath(user_matlab_path); savepath;