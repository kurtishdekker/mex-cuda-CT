function [filteredSinogram, filterKernel] = preFilterSinogram(sino,geom,angles,filter,frequencyCutoff)
% This function applies weighting and filtering to 2D projections (cone,
% fan3D or par3D) in order to perform FBP / FDK reconstruction
%
% Inputs:
%   sino = sinogram projections
%   geom = struct containing the geometry parameters
%       -> geom.type = 'cone','fan3D','par3D' (string)
%       -> geom.SAD = source-axis distance (only for cone / fan3D datasets)
%
%   filter = filter to apply ('Ramp', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann', or 'Blackman')
%   frequencyCutoff = cutoff frequency (normalized between [0 1]) for the filtering step
%
% Outputs:
%   filteredSinogram = filtered data ready for backprojection
%   filterKernel = the applied filter (in Fourier domain)
% 
% Author: Kurtis Dekker  (LHSC) 
% History:
%           08-04-2016 - khd initial version
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input checking / selection of geometry
%check number of inputs
if nargin<4
   error('Not enough arguments in filter');
end
N = size(sino,1);

%% distance weighting (geometry dependent)
switch lower(geom.type)

    case 'par3d'
        %no geometry weighting required
    case 'fan3d'
        SAD = geom.SAD;
        [U V] = ndgrid(1:size(sino,1), 1:size(sino,2));
        U = U-(size(sino,1)-1)/2;
        Distance = SAD./sqrt(SAD^2 + U.^2);
        sino = bsxfun(@times,sino,Distance);
    case 'cone3d'
    %cone beam weighting
        SAD = geom.SAD;
        [U V] = ndgrid(1:size(sino,1), 1:size(sino,2));
        U = U-(size(sino,1)-1)/2;
        V = V-(size(sino,2)-1)/2;
        Distance = SAD./sqrt(SAD^2 + U.^2 + V.^2);
        sino = bsxfun(@times,sino,Distance);
end

%% Filtering

%generate filter
H = designFilter(filter, N, frequencyCutoff);

filteredSinogram = zeros(size(sino,1),size(sino,2),size(sino,3));
% Filtering
parfor i = 1:size(sino,3)
    tmp = sino(:,:,i);
    tmp(length(H),1) = 0; %zeropad to 2N
   % tmp = padarray(tmp,[N/2,0]); %zeroPad to 2N
    
    tmp = ifft(bsxfun(@times,fft(tmp),H),'symmetric');
    tmp(N+1 : end, :) = []; %trim zero-padding
    %tmp = tmp(N/2+1 : (3*N/2),:,:); %trim zero padding
    if i>1 
        tmp = tmp .* (angles(i)-angles(i-1))/4;
    else
        tmp = tmp .* (angles(i+1)-angles(i))/4;
    end
    
    filteredSinogram(:,:,i) = (tmp);

end
filterKernel = H;
 filteredSinogram = single(filteredSinogram);
%----------------------------------------------------------------------

%% filter design function
function filt = designFilter(filter, len, d)
% Returns the Fourier Transform of the filter which will be
% used to filter the projections
%
% INPUT ARGS:   filter - either the string specifying the filter
%               len    - the length of the projections
%               d      - the fraction of frequencies below the nyquist
%                        which we want to pass
%
% OUTPUT ARGS:  filt   - the filter to use on the projections


order = max(64,2^nextpow2(2*len));

if strcmpi(filter, 'none')
    filt = ones(1, order)';
    return;
end

% First create a bandlimited ramp filter (Eqn. 61 Chapter 3, Kak and
% Slaney) - go up to the next highest power of 2.

n = 0:(order/2); % 'order' is always even. 
filtImpResp = zeros(1,(order/2)+1); % 'filtImpResp' is the bandlimited ramp's impulse response (values for even n are 0)
filtImpResp(1) = 1/4; % Set the DC term 
filtImpResp(2:2:end) = -1./((pi*n(2:2:end)).^2); % Set the values for odd n
filtImpResp = [filtImpResp filtImpResp(end-1:-1:2)]; 
filt = 2*real(fft(filtImpResp)); 
filt = filt(1:(order/2)+1);

w = 2*pi*(0:size(filt,2)-1)/order;   % frequency axis up to Nyquist

if ~isempty(regexpi(filter,'zeng'))
    kval = str2num(filter(end-3 : end));
    if isempty(kval)
        kval = inf;
    end
    filt(2:end) = filt(2:end) .* (1 - (1 - 0.5*.00525./filt(2:end)).^kval);
else
switch lower(filter)
    case 'ramp'
        % Do nothing
    case 'shepp-logan'
        % be careful not to divide by 0:
        filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)));
    case 'cosine'
        filt(2:end) = filt(2:end) .* cos(w(2:end)/(2*d));
    case 'hamming'
        filt(2:end) = filt(2:end) .* (.54 + .46 * cos(w(2:end)/d));
    case 'hann'
        filt(2:end) = filt(2:end) .*(1+cos(w(2:end)./d)) / 2;
    otherwise
        error(message('images:iradon:invalidFilter'))
end
end

filt(w>pi*d) = 0;                      % Crop the frequency response
filt = [filt' ; filt(end-1:-1:2)'];    % Symmetry of the filter
%----------------------------------------------------------------------