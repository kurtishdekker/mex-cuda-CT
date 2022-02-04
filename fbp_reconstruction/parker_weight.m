function [weightedSino] = parker_weight(sino,angles,geom, q_value)
% This function applies Parker weighting to 2D projections (cone,
% fan3D or par3D) in order to perform FBP / FDK reconstruction with
% short-scan data (i.e. less than 2*pi)
%
% Inputs:
%   sino = sinogram projections
%   angles = vector containing the projection angles in radians
%   geom = struct containing the geometry parameters
%       -> geom.type = 'cone3d','fan3d','par3d' (string)
%       -> geom.SAD = source-axis distance (only for cone / fan3D datasets)
%   q_value = q term to weight the trapezoids for modified parker weights.
%       -> q = 1 corresponds to original Parker weighting as in 1982 paper
%       -> q = 0.1 is the suggested value from Wesarg et al 2002.
%   
%
% Outputs:
%   weightedSino = parker-weighted sinogram data ready for FBP/FDK
% 
% References:
%   Wesarg et al., "Parker weights revisited", Med. Phys. 29 (3), 2002
%
% Author: Kurtis H. Dekker, PhD  (CCSEO) 
% History:
%           02-02-2022 - khd initial version
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check geometry and calculate alpha values for projection rays
if strcmpi(geom.type, 'par3d') %if parallel beam, then alpha = 0 for all rays in a projection
    alpha = zeros(1,size(sino,2));
else %fan/cone beam, alpha value calculated. note: we have assumed flat detector fan beam CT
    alpha = atan(((-size(sino,2)/2 + 0.5) : (size(sino,2)/2-.5))/geom.SAD);
end

%% Check scan angular extent and check fan/cone angle
diff_angles = diff(angles);

if(all(diff_angles >=0))
    alpha = -alpha;
end

delta = abs(alpha(end)-alpha(1))/2;
scanExtent = abs(angles(end)-angles(1));

if scanExtent >= 2*pi
    warning('Computing Parker weights for angle equal or greater than 2*pi');
end
if scanExtent <pi+2*delta
    warning('Scan angle extend less than pi + fan/cone angle. Parker weights not required as no redundant data exists');
end

switch q_value
    case 1
        disp('q = 1, corresponds to original Parker weighting');
    otherwise
        disp (['q = ' num2str(q_value) ', modified Parker weights (Wesarg et al 2002)']);
end

epsilon = max(scanExtent-(pi+2*delta),0);


%% loop through sinogram and apply Parker / modified Parker weights

for i = 1:size(sino,3)
    beta = abs(angles(i)-angles(1));
    
    parker_window   =   0.5 * (S_function(beta ./ b2_function(alpha,delta,epsilon,q_value)-0.5) ...
                        + S_function((beta-2*delta+2*alpha-epsilon) ./ b2_function(alpha,delta,epsilon,q_value) + 0.5)...
                        - S_function((beta-pi+2*alpha) ./ b2_function(-alpha,delta,epsilon,q_value)-0.5) ...
                        - S_function((beta-pi-2*delta-epsilon) ./ b2_function(-alpha,delta,epsilon,q_value)+0.5));
      
    parker_window=single(parker_window');
    sino(:,:,i) = sino(:,:,i) .* repmat(parker_window,[1,size(sino,2)]);
    

end
weightedSino = sino;
end

%% S term from Wesarg et al 2002 (window function)
function window = S_function(beta) 

window=zeros(size(beta));
window(beta<=-0.5)=0;
window(abs(beta)<0.5)=0.5*(1+sin(pi*beta(abs(beta)<0.5)));
window(beta>=0.5)=1;

end

%% B term from Wesarg et al 2002
function b1_value = b1_function(alpha,delta,epsilon) %B term from Wesarg et al 2002
    b1_value = 2 * delta - 2 * alpha + epsilon;
end

%% b term from Wesarg et al 2002
function b2_value = b2_function(alpha,delta,epsilon,q_value) %b term from Wesarg et al 2002
    b2_value = q_value * b1_function(alpha,delta,epsilon);
end