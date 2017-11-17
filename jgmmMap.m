%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mapping routine for the following graphical model:
%
%    m
%  / | \
% y->x->z
% |     ^
% |     |
%  -----
% with m is unobserved and z is partially observed
% Thomas Hueber - Laurent Girin - Xavi Alameda
% CNRS/GIPSA-lab - 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [y_mse,varargout] = jgmmMap(z,jgmmParam)

M = jgmmParam.M;
N = size(z,2);
nDimxy = jgmmParam.nDimx+jgmmParam.nDimy;
nDimxyz = jgmmParam.nDimx+jgmmParam.nDimy+jgmmParam.nDimz;
y_ind = jgmmParam.nDimx+1:nDimxy;
z_ind  =nDimxy+1:nDimxyz;

% Caculate responsabilities for each input frame (posteriors)
beta = gmmCalculatePosterior(z,jgmmParam.xyz_priors,jgmmParam.xyz_mu(z_ind,:),jgmmParam.xyz_sigma(z_ind,z_ind,:));

% MSE-mapping
y_mse = zeros(jgmmParam.nDimy,N);
for n=1:N
    tmp = zeros(jgmmParam.nDimy,1);
    for m = 1:M
        tmp = tmp + beta(n,m).*(jgmmParam.xyz_mu(y_ind,m) + jgmmParam.xyz_sigma(y_ind,z_ind,m)*inv(jgmmParam.xyz_sigma(z_ind,z_ind,m))*(z(:,n)-jgmmParam.xyz_mu(z_ind,m)));
    end
    y_mse(:,n) = tmp;
end

if nargout>0
    varargout{1}=beta;
end

%% END of cgmmMap FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%