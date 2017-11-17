% Mapping routine for the Integrated Cascaded Gaussian Mixture Regression
% (IC-GMR) (based on MSE criterion)
%
%    m
%  / | \
% z<-x<-y
%
% defined in:
% --------
% T. Hueber, L. Girin, X. Alameda-Pineda, and G. Bailly,
% ?Speaker-adaptive acoustic-articulatory inversion using cascaded Gaussian mixture regression,?
% IEEE Transactions on Audio, Speech and Language Processing, July 2015.
% --------
%
% INPUT:
% z: [nDimz N] matrix of N input observations
% cgmmParam stuct containing IC-GMR model parameters, estimated using the cgmmTrain function
%
% OUTPUT:
% y_mse: [nDimy N] matrix of N estimated output observations using IC-GMR (MSE criterion)
%
% See README.txt for more information about the C-GMR Matlab package
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [y_mse,varargout] = cgmmMap(z,cgmmParam)

M = cgmmParam.M;
N = size(z,2);

% Caculate responsabilities for each input frame (posteriors)
beta = gmmCalculatePosterior(z,cgmmParam.xyz_priors,cgmmParam.f,cgmmParam.P);

% MSE-mapping
y_mse = zeros(cgmmParam.nDimy,N);
for n=1:N
    tmp = zeros(cgmmParam.nDimy,1);
    for m = 1:M
        tmp = tmp + beta(n,m).*(cgmmParam.A_star(:,:,m)*cgmmParam.C_star(:,:,m)*z(:,n)+cgmmParam.A_star(:,:,m)*cgmmParam.d_star(:,m)+cgmmParam.b_star(:,m));
    end
    y_mse(:,n) = tmp;
end


if nargout>0
    varargout{1}=beta;
end

%% END of cgmmMap FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%