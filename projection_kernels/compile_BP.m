%compile backprojection CBCT code
%kurtis h. dekker (lrcp, uwo)

%creates CUDAmex_backprojection.mexw64


!hostname > hostname.txt
hostname = textread('hostname.txt','%s');
delete('hostname.txt');

if strcmpi(hostname,'khd-pc')

    nvccCommandLine = 'nvcc -c CUDAmex_BP.cu -o CUDAmex_BP.o -O3 -gencode=arch=compute_61,code=sm_61 --ptxas-options=-v -I"C:\Program Files\MATLAB\R2016b\extern\include" -l"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"';
    status = system(nvccCommandLine);
    if status < 0
        error 'Error invoking nvcc';
    end
    mex('-largeArrayDims', 'CUDAmex_BP.o', '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64', '-lcudart');

    
elseif strcmpi(hostname,'optics-khd')

    nvccCommandLine = 'nvcc -c CUDAmex_BP.cu -o CUDAmex_BP.o -O3 -gencode=arch=compute_52,code=sm_52 --ptxas-options=-v -I"C:\Program Files\MATLAB\R2015a\extern\include" -l"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\lib" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin"';
    status = system(nvccCommandLine);
    if status < 0
        error 'Error invoking nvcc';
    end
    mex('-largeArrayDims', 'CUDAmex_BP.o', '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\lib\x64', '-lcudart');

    
else
    
    nvccCommandLine = 'nvcc -c CUDAmex_BP.cu -o CUDAmex_BP.o -O3 -gencode=arch=compute_35,code=sm_35 --ptxas-options=-v -I"C:\Program Files\MATLAB\R2015a\extern\include" -l"C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\lib"';
    status = system(nvccCommandLine);
    if status < 0
        error 'Error invoking nvcc';
    end
    mex('-largeArrayDims', 'CUDAmex_BP.o', '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\lib\x64', '-lcudart');

end