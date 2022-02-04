%% generate phantom

ph = single(repmat(phantom('Modified Shepp-Logan',384),[1 1 384]));
ph(ph<0)=0;
ph(:,:,1:70) = 0;
ph(:,:,end-70:end)=-0;

%% set up scan geometry
nProjections = 512;

SAD=1000;
 angles = single(linspace(0,pi + 2*atan(192/SAD),nProjections));
%angles = single(linspace(0,2*pi ,nProjections));
parker_q = 1;

%% parallel beam
geomFlag = 0; %0 = parallel-beam, 1 = fan-beam, 2 = cone-beam

% compute forward projections
fp_par = CUDAmex_FP(ph, SAD, angles, nProjections, geomFlag);

% filter sinogram data
filteredSino = single(preFilterSinogram(fp_par, struct('type','par3d','SAD',SAD),angles,'hamming',1, parker_q));

% reconstruct via backprojection
recon_par = CUDAmex_BP(filteredSino, SAD, angles, geomFlag);
%% fan beam
geomFlag = 1; %0 = parallel-beam, 1 = fan-beam, 2 = cone-beam

% compute forward projections
fp_fan = CUDAmex_FP(ph, SAD, angles, nProjections, geomFlag);

% filter sinogram data
filteredSino = single(preFilterSinogram(fp_fan, struct('type','fan3d','SAD',SAD),angles,'hamming',1, parker_q));

% reconstruct via backprojection
recon_fan = CUDAmex_BP(filteredSino, SAD, angles, geomFlag);
%% Cone Beam

geomFlag = 2; %0 = parallel-beam, 1 = fan-beam, 2 = cone-beam

% compute forward projections
fp_cone = CUDAmex_FP(ph, SAD, angles, nProjections, geomFlag);

% filter sinogram data
filteredSino = single(preFilterSinogram(fp_cone, struct('type','cone3d','SAD',SAD),angles,'hamming',1, parker_q));

% reconstruct via backprojection
recon_cone = CUDAmex_BP(filteredSino, SAD, angles, geomFlag);