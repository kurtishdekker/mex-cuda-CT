%% generate cylindrical phantom
[x y] = meshgrid(-127.5 : 127.5, -127.5 : 127.5);
ph2 = (x.^2 + y.^2 < 94.^2);
ph = repmat(ph2,[1 1 256])*.02;
ph = single(ph);

%% specify scan geometry
projsize=[256,256];
nProjections=512;
SAD = 1000; %not needed for parallel beam, but a placeholder

% set up points array (parallel beam, but specified via "general3D" option)
x1 = ones(projsize)* (-1000);
x1 = single(x1);
x2 = -x1;
y1 = -127.5 : 127.5;
y1=y1';
y1 = repmat(y1,[1,projsize(2)]);
y1=single(y1);
y2=y1;
z1 = -127.5 : 127.5;
z1 = repmat(z1,[projsize(1),1]);
z1 = single(z1);
z2=z1;
clear pts;

% assign coordinates to points array
pts(:,:,1) = x1; pts(:,:,2) = y1; pts(:,:,3) = z1;
pts(:,:,4) = x2; pts(:,:,5) = y2; pts(:,:,6) = z2;


%% specify projection angles
angles = single(linspace(0,2*pi,nProjections));

%% forward projections
tic; fp_par = CUDAmex_FP(ph,SAD, angles, nProjections); toc;

tic; fp_general = CUDAmex_general3D_FP(ph,angles,pts,nProjections); toc;
%% reconstruct
filt = single(preFilterSinogram(fp_par,struct('type','par3d','SAD',1000),angles,'hamming',1));
tic; bp_par = CUDAmex_BP(filt,inf,angles,0); toc;
tic; bp_general = CUDAmex_general3D_BP(filt,angles,pts, 256,256); toc; %yields discretization artifacts unless super-sampling of rays is done

