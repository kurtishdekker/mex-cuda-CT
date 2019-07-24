# mex-cuda-CT

## preamble
GPU-accelerated forward and back projection operators for CT reconstruction.
Includes code for parallel-beam, fan-beam, and cone-beam geometries. 
Additionally, a "general-3D" geometry option exists, where non-standard ray-paths can be specified.


The purpose of this toolbox is not to provide a large number of implemented reconstruction algorithms. See the ASTRA<sup>1</sup> or TIGRE<sup>2</sup> toolboxes for a more complete set. 
Rather, the goal here is to provide the fundamental building blocks - the projection operators - without overwhelming potential users. That said, a simple filtered backprojection algorithm (FBP)<sup>3</sup> and a TV-minimization based iterative algorithm (OSC-TV)<sup>4</sup> are provided

## Usage Example - simple FBP
```MATLAB
% generate phantom image
ph = single(phantom3(256)*.02);

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
