# mex-cuda-CT

GPU-accelerated forward and back projection operators for CT reconstruction.

Includes code for parallel-beam, fan-beam, and cone-beam geometries. 
Additionally, a "general-3D" geometry option exists, where non-standard ray-paths can be specified.


The purpose of this toolbox is not to provide a large number of implemented reconstruction algorithms. See the ASTRA [CITE] or TIGRE [CITE] toolboxes for a more complete set. 
Rather, the goal here is to provide the fundamental building blocks - the projection operators - without overwhelming potential users. That said, a simple filtered backprojection algorithm (FBP)[CITE] and a TV-minimization based iterative algorithm (OSC-TV) [CITE] are provided
