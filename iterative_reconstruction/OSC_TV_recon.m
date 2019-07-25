function recon = OSC_TV_recon(pre,post, scanAngles, geom, numIter,TV_constant,numSubsInit,numSubsFinal,pVal)
%OSC_TV_CBCT.M - gpu reconstruction with OSC-TV (matenine, 2015, Laval U)
%
%Inputs:
%       pre - reference scan (MxNxNproj, eg. 480x640x1024)
%       post - data scan
%       geom - the structure specifying fan, cone, or
%       parallel-beam CT geometry and SAD if needed
%       scanAngularExtent - rotational extent of the CBCT scan (radians)
%       numIter - number of iterations to perform
%       tv_constant - regularization constant. Typical value is 0.02 or
%       0.05 (Matenine et al 2015)
%
%Output:
%       recon - final reconstruction dataset. Size will be determined by
%       projection size
%
%Dependencies:
%       NVIDIA GPU with sufficient memory for recon grid
%       CUDA toolbox v6.0 or higher
%
%Created:  April 10 2017 by Kurtis H. Dekker
%Modified: July 25 2019 by KHD
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = ones(size(post,1),size(post,1),size(post,2),'single').*1e-10; %adjust size here
nProj = size(post,3);
epsilon=1e-6;
pre(pre<epsilon)=epsilon;
post(post<epsilon)=epsilon;

[x,y] = meshgrid(1:size(post,1),1:size(post,1));
c = ((x-size(post,1)/2).^2 + (y-size(post,1)/2).^2) < (size(x,1)/2)^2;
pre2=pre; post2=post; pre2(1) = 0; post2(1) = 0;


for k = 1:numIter
    mu_old = mu; mu_old(1) = 0;
    
    %calculate number and order of subsets for iteration "k"
    numSubsets = calcNumSubsets(numSubsInit,numSubsFinal,pVal,numIter,k);
    subsetPicks = calcSubsetOrder(numSubsets);
     
    if strcmpi(geom.type,'par3d')
        CUDAmex_oscIter(mu, pre, post, numSubsets, subsetPicks, scanAngles,geom.SAD,0);
    elseif strcmpi(geom.type,'fan3d')
        CUDAmex_oscIter(mu, pre, post, numSubsets, subsetPicks, scanAngles, geom.SAD,1);
    elseif strcmpi(geom.type,'cone')
        CUDAmex_oscIter(mu,pre,post,numSubsets,subsetPicks,scanAngles, geom.SAD,2);
    else
        error('Invalid Geometry Selection');
    end
       
    
    pre=pre2; post=post2; pre(1) = 0; post(1) = 0;
   
    if k > 1
        mu_old = (mu_old - mu);
        mu_old(isnan(mu_old))=0;
        d_a = sqrt(sum(mu_old(:).^2));
        mu = CUDAmex_TVmin_3D(mu,TV_constant*d_a); %call mex for Tv-min, first divide TV-const  by d_a
    end
    figure(1000); imagesc(mu(:,:,round(size(mu,3)/2))); axis equal; axis tight; title(['iteration: ' num2str(k)]); pause(0.01);
end


function numSubsets = calcNumSubsets(sInit,sFinal,pVal,maxIters,iterationNumber)

numSubsets = round((sInit - sFinal)/(maxIters-1)^pVal * (maxIters  - iterationNumber)^pVal + sFinal);

function subsetPicks = calcSubsetOrder(numSubsets)
    subsetNums = 1:numSubsets;
    subsetInd_current = randperm(numel(subsetNums),1);
    subsetPicks = [];
    for s = 1:numSubsets
        %subset choosing logic
        subsetNums_lower = subsetNums(subsetNums < subsetInd_current);
        subsetNums_higher = subsetNums(subsetNums > subsetInd_current);
        
        if s > 1
            if numel(subsetNums_lower) > numel(subsetNums_higher)
                subsetInd_current = subsetNums_lower(ceil(numel(subsetNums_lower)/2));
                subsetNums_lower(subsetNums_lower == subsetInd_current) = [];
                subsetNums = [subsetNums_lower, subsetNums_higher];
            else
                subsetInd_current = subsetNums_higher(ceil(numel(subsetNums_higher)/2));
                subsetNums_higher(subsetNums_higher == subsetInd_current) = [];
                subsetNums = [subsetNums_lower, subsetNums_higher];
            end
        end
        subsetPicks(end+1) = subsetInd_current;
    end
    subsetPicks = int32(subsetPicks-1); %cast as int32 prior to passing into MEX