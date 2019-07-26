%% phantom

 ph = zeros([256 256 256],'single');
 [x y] = meshgrid(-127.5 : 127.5,-127.5 : 127.5);
 mask = ((x.^2 + y.^2) <=(114^2));

 ph = repmat(single(mask),[1 1 size(ph,3)])*.002;


field = gaussian2d(32,4);
field = field./max(field(:)) * 0.01 + 0.002;
% ph((362:425),256-31 : 256+32,:) = repmat(field(1:end-1,1:end-1),[1 1 size(ph,3)]);
ph((182:213),128-15 : 128+16,:) = repmat(field(1:end-1,1:end-1),[1 1 size(ph,3)]);

 %% parameters
nProj = 512;
angles = single(linspace(0,2*pi,nProj));
SAD = 1000;

g_par.type = 'par3d';
g_par.SAD = SAD;
geomFlag = 0;
filter = 'hamming';
filterCutoff = 0.8;


%% FP
fp_par = CUDAmex_FP(ph,SAD,angles, nProj,geomFlag) ; 
pre = 1e5.*ones(size(fp_par),'single');
post = pre .* exp(-fp_par);


%% FBP recon
sino = -log(post./pre); sino(isinf(sino)) = 0; sino(isnan(sino))=0;
filt_par = single(preFilterSinogram(fp_par,g_par,angles,filter,filterCutoff));
bp_par = CUDAmex_BP(filt_par,SAD,angles,geomFlag);


%% OSC-TV recon
tvConst = 0.05;
pVal = 0.5;
nIter = 10;
nSubsInit = 64;
nSubsFinal = 4;
tic;
recon = OSC_TV_recon(pre, post,angles,g_par,nIter,tvConst,nSubsInit,nSubsFinal,pVal);
toc;
%% plot profiles
figure; 
plot(squeeze(ph(:,128,128)))
hold on;
plot(squeeze(bp_par(:,128,128)),'-.');
plot(squeeze(recon(:,128,128)),'-.');
legend('truth','FBP','OSC-TV')

