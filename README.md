# mex-cuda-CT

## preamble
This repository provides GPU-accelerated forward and back projection operators for CT reconstruction, coded in CUDA-C and interfaced with MATLAB through the generation of MEX files.

Projection operators are available for parallel-beam, fan-beam, and cone-beam geometries. 
Additionally, a "general-3D" geometry option exists, where non-standard ray-paths can be specified.

The purpose of this toolset is not to provide a large number of implemented reconstruction algorithms. See the [ASTRA<sup>1</sup>][4] or [TIGRE<sup>2</sup>][3] toolboxes for a more complete set. 
Rather, the goal here is to provide the fundamental building blocks - the projection operators - without overwhelming potential users with a myriad of options. That said, a sinogram filtering function for filtered backprojection reconstruction (FBP)<sup>3</sup>, and a TV-minimization based iterative reconstruction algorithm (OSC-TV)<sup>4</sup> are provided as a jumping-off point for users who want to get straight to reconstructing their projection data.

The intended user of these tools is one who wants a direct programmatic interface to GPU-accelerated CT operations and does not need (or, perhaps, want) a standalone application or GUI.

## installation
Specific installation instructions are provided only for Microsoft Windows 7/10. An enterprising Linux user should be able to compile the tools, but I have not been able to try this myself. 

[Windows Installation Instructions](documentation/installation.md)


## Usage Example - simple FBP
This example generates cone-beam projections from a uniform cylindrical phantom, and performs a basic FDK reconstruction (Filtered backprojection). A much more interesting phantom, the 3D Shepp-Logan head phantom, can be generated using a nice [MATLAB function by Matthias Schabel][2].

```MATLAB
% generate simple cylinder phantom
[x y] = meshgrid(-191.5 : 191.5,-191.5 : 191.5);
mask = ((x.^2 + y.^2) <=(140^2));
ph = repmat(single(mask),[1 1 384])*0.02;

% set up scan geometry
nProjections = 512;
angles = single(linspace(0,2*pi,nProjections));
SAD=1000;
geomFlag = 2; %0 = parallel-beam, 1 = fan-beam, 2 = cone-beam

% compute forward projections
fp = CUDAmex_FP(ph, SAD, angles, nProjections, geomFlag);

% filter sinogram data
filteredSino = single(preFilterSinogram(fp, struct('type','cone3d','SAD',SAD),angles,'hamming',1);

% reconstruct via backprojection
recon = CUDAmex_BP(filteredSino, SAD, angles, geomFlag);
```


## references
<sup>1</sup>W. van Aarle et al., “The ASTRA Toolbox: A platform for advanced algorithm development in electron tomography,” Ultramicroscopy, vol. 157, pp. 35–47, Oct. 2015.

<sup>2</sup>A. Biguri, M. Dosanjh, S. Hancock, and M. Soleimani, “TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction,” Biomed. Phys. Eng. Express, vol. 2, no. 5, p. 055010, 2016.

<sup>3</sup>A. C. Kak and M. Slaney, Principles of Computerized Tomographic Imaging. SIAM, 2001.

<sup>4</sup>D. Matenine, Y. Goussard, and P. Després, “GPU-accelerated regularized iterative reconstruction for few-view cone beam CT,” Medical Physics, vol. 42, no. 4, pp. 1505–1517, Apr. 2015.

## licensing

The tools here are provided as-is under the [BSD License][1].

[1]:LICENSE
[2]:https://www.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom
[3]:https://github.com/CERN/TIGRE
[4]:https://www.astra-toolbox.com/
