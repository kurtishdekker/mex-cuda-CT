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

2. Install the CUDA toolkit
   
   The code has been tested using CUDA v7.0 and v8.0, however no issues are anticipated with using the latest version.

3. Install Visual Studio
   
   **C++ is required.**
   The tools have been successfully compiled using both Visual Studio
