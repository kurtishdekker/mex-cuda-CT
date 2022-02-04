%% phantom

ph = single(repmat(phantom('Modified Shepp-Logan',256),[1 1 256])*.002);
ph(ph<0)=0;
ph(:,:,1:40) = 0;
ph(:,:,end-40:end)=-0;

 %% parameters
nProj = 512;

SAD = 1000;
 angles = single(linspace(0,pi + 2*atan(128/SAD),nProj));
geom.type = 'cone3d';
geom.SAD = SAD;
geomFlag = 2; 

filter = 'ramp';
filterCutoff = 0.8;
parker_q = 0.1;


%% FP
fp = CUDAmex_FP(ph,SAD,angles, nProj,geomFlag) ; 
pre = 1e6.*ones(size(fp),'single');
post = pre .* exp(-fp);

for i=1:nProj
    pre(:,:,i) = single(imnoise(pre(:,:,i)*1e-6,'poisson')*1e6);
    post(:,:,i) = single(imnoise(post(:,:,i)*1e-6,'poisson')*1e6);
end



%% FBP recon
sino = -log(post./pre); sino(isinf(sino)) = 0; sino(isnan(sino))=0;
filt_par = single(preFilterSinogram(sino,geom,angles,filter,filterCutoff,parker_q));
bp = CUDAmex_BP(filt_par,SAD,angles,geomFlag);



%% OSC-TV recon
tvConst = 0.1;
pVal = 0.2;
nIter = 10;
nSubsInit = 64;
nSubsFinal = 2;
tic;
recon = OSC_TV_recon(pre, post,angles,geom,nIter,tvConst,nSubsInit,nSubsFinal,pVal);
toc;


%% plot profiles
figure; 
plot(squeeze(ph(:,128,128)))
hold on;
plot(squeeze(bp(:,128,128)),'-.');
plot(squeeze(recon(:,128,128)),'-.');
legend('truth','FBP','OSC-TV')


%% 3d viewer
imagine(ph,bp, recon);
