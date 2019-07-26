# installation instructions

## Windows

### requirements
1. [MATLAB](https://www.mathworks.com/products/matlab.html)
2. [Visual Studio (Community or Professional edition)](https://visualstudio.microsoft.com/)
3. [An NVIDIA GPU with CUDA support and compute capability > 3.0](https://developer.nvidia.com/cuda-gpus)
4. [NVIDIA CUDA toolkit v7.0 or higher](https://developer.nvidia.com/cuda-toolkit)

### simple installation instructions
1. Install MATLAB, Visual Studio, and the CUDA toolkit.
2. Change the appropriate variables in `compile.m` and run the script.

### detailed instructions
1. Install MATLAB

2. Install Visual Studio

   The tools have been successfully compiled using Visual Studio 2012 (v11.0) and Visual Studio 2015 (v14.0). Newer versions should not pose a problem provided they are compatible with your version of MATLAB (for mex-file generation) and CUDA. 
   
   *C++ is required.* Ensure that the option to install the C++ tools is selected during installation of Visual Studio.

3. Ensure that MATLAB is configured to use the Visual Studio compiler for mex file generation

   In MATLAB, run `mex -setup -v' and ensure that the Visual C++ compiler is found. If not, it indicates that either Visual Studio is not installed or the C++ tools were not installed. 

   Make sure that Visual C++ is selected for use by MATLAB's `mex` command.


4. Install the CUDA toolkit
   
   The code has been tested using CUDA v7.0 and v8.0, however no issues are anticipated with using the latest version.
   
   [Download](https://developer.nvidia.com/cuda-downloads) and install the toolkit. An [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) specific to the CUDA tools is provided by NVIDIA. 
   
   After installing CUDA, check that you have the CUDA tools on your $PATH variable, or alternatively that you have a CUDA_PATH environment variable defined. This can be accomplished through `Control Panel -> System -> Advanced System Settings -> Environment Variables`.
   
5. Download mex-cuda-CT

   Either use git (`git clone https://github.com/kurtishdekker/mex-cuda-CT.git`) or [download the zipped repository](https://github.com/kurtishdekker/mex-cuda-CT/archive/master.zip).
   
6. Compile the tools

   In MATLAB, navigate to the top level of the `mex-cuda-CT` directory you downloaded in step 5. Edit the `compile.m` file. You will need to change the following lines:
   
   `vs_path = 'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\'` should be changed to match your installation of Visual Studio
   
   `gpu_compute = '-gencode=arch=compute_XX,code=sm_XX'` needs to have the `XX`s replaced with the [compute capability](https://developer.nvidia.com/cuda-gpus) for your specific GPU. For example, to compile for a GPU with compute capability of `6.1`, replace `XX` with `61` (do not include the `.`).
   
   Once this has been done, run the `compile.m` script. If no errors occur, you should have compiled `.mex` files located in your MATLAB user directory in a subfolder called `mex-cuda-CT`, which should be automatically added to your MATLAB `path` variable.

7. Test functionality

   The functionality of the forward and back projection operators and FBP reconstruction code can be tested using `examples/projection_FBP_test.m` (for standard geometry) and `examples/general3D_projection_test.m` for the general3D projectors. 

   The functionality of OSC-TV reconstruction code can be tested using `examples/OSC_TV_test.m`.

