%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training routine for the following graphical model:
%
%    m
%  / | \
% y->x->z
% |     ^
% |     |
%  -----
% with m is unobserved and z is partially observed
% p(x,y) can be used for initialization
%
% [jgmmParam, log_likelihood_curve] = train_jgmm(x,y,z,xy_priors,xy_mu,xy_sigma,maxIterEM)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% T. Hueber,  PhD. CNRS/GIPSA-Lab
% L. Girin,  Prof. GIPSA-Lab
% X. Alameda, PhD. UNITN
% Copyleft 2015
%
% This file is part of C-GMM.
%
% C-GMM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% C-GMM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

function jgmmParam = jgmmTrain(x,y,z,xy_priors,xy_mu,xy_sigma,iterEM)

% Set variables
numberOfComponents = size(xy_mu,2);
No = size(z,2);
N = size(x,2);

nDimx = size(x,1);
nDimy = size(y,1);
nDimz = size(z,1);
nDimxy = nDimx+nDimy;
nDimxyz = nDimx+nDimy+nDimz;

xy_ind = 1:(nDimxy);
z_ind = nDimxy+1:nDimxyz;

xy = [x;y];
xyz = [x(:,1:No);y(:,1:No);z];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize p(x,y,z) by the suboptimal procedure %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate beta=p(m|[x y])=p([x y]|m)p(m)
beta = gmmCalculatePosterior(xy,xy_priors,xy_mu,xy_sigma);
xyz_priors = xy_priors;
xyz_mu = zeros(nDimxyz,numberOfComponents);
xyz_sigma = zeros(nDimxyz,nDimxyz,numberOfComponents);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization using a full linear model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Declare variables
S = sum(beta); So = sum(beta(1:No,:));
S_x = x*beta;
S_y = y*beta;
% Declare zero-valued variables
S_z = zeros(nDimz,numberOfComponents);
S_xx = zeros(nDimx,nDimx,numberOfComponents);
S_zx = zeros(nDimz,nDimx,numberOfComponents);
S_xy = zeros(nDimx,nDimy,numberOfComponents);
S_yy = zeros(nDimy,nDimy,numberOfComponents);
S_zy = zeros(nDimz,nDimy,numberOfComponents);
% Declare regularizers
reg_cov_y = 1e-7*eye(nDimy);
reg_cov_x = 1e-7*eye(nDimx);
reg_cov_z = 1e-7*eye(nDimz);

for m = 1:numberOfComponents
   % Sufficient statistics
    S_z(:,m) = z*beta(1:No,m);
    S_xx(:,:,m) = x.*repmat(beta(:,m),1,nDimx)'*x';
    S_zx(:,:,m) = z(:,1:No).*repmat(beta(1:No,m),1,nDimz)'*x(:,1:No)';
    S_zy(:,:,m) = z(:,1:No).*repmat(beta(1:No,m),1,nDimz)'*y(:,1:No)';
    S_xy(:,:,m) = x.*repmat(beta(:,m),1,nDimx)'*y';
    S_yy(:,:,m) = y.*repmat(beta(:,m),1,nDimy)'*y';
    
    % Parameters
    par_l.mu_y = S_y(:,m)./repmat(S(m),nDimy,1);
    par_l.cov_y = (1./S(m)).*(y-repmat(par_l.mu_y,1,N)).*repmat(beta(:,m),1,nDimy)'*(y-repmat(par_l.mu_y,1,N))'+reg_cov_y;
    par_l.lin_xy = (S_xy(:,:,m)-(1./S(m)).*S_x(:,m)*S_y(:,m)')/(S_yy(:,:,m)-(1./S(m)).*S_y(:,m)*S_y(:,m)');
    par_l.ind_x = (1./S(m)).*(S_x(:,m)-par_l.lin_xy*S_y(:,m));
    par_l.cov_x = (1./S(m)).*(x-par_l.lin_xy*y-repmat(par_l.ind_x,1,N)).*repmat(beta(:,m),1,nDimx)'*((x-par_l.lin_xy*y-repmat(par_l.ind_x,1,N)))'+reg_cov_x;
    
    %par_l.lin_zx = (S(m)/So(m))*(S_zx(:,:,m)-(1./S(m)).*S_z(:,m)*S_x(:,m)')/(S_xx(:,:,m)-(1./S(m)).*S_x(:,m)*S_x(:,m)');
    %par_l.lin_zy = (S(m)/So(m))*(S_zy(:,:,m)-(1./S(m)).*S_z(:,m)*S_y(:,m)')/(S_yy(:,:,m)-(1./S(m)).*S_y(:,m)*S_y(:,m)');
    %par_l.ind_z = (1./S(m)).*(S_z(:,m)*S(m)/So(m)-par_l.lin_zx*S_x(:,m)-par_l.lin_zy*S_y(:,m));
    %par_l.cov_z = (1./So(m)).*( (z(:,1:No)-par_l.lin_zx*x(:,1:No)-par_l.lin_zy*y(:,1:No)-repmat(par_l.ind_z,1,No)).*repmat(beta(1:No,m),1,nDimz)'*(z(:,1:No)-par_l.lin_zx*x(:,1:No)-par_l.lin_zy*y(:,1:No)-repmat(par_l.ind_z,1,No))' )+reg_cov_z;
    
    % an other way to initialize the model
    par_l.lin_zx =  eye(nDimz,nDimx);
    par_l.lin_zy = zeros(nDimz,nDimy);
    par_l.ind_z = zeros(nDimz,1);
    par_l.cov_z = par_l.cov_x;
    
    
    % Convert parameters
    par_j = gaussian_par_l2j(par_l);
    
    % Store
    xyz_mu(:,m) = par_j.mu;
    xyz_sigma(:,:,m) = par_j.cov;%+1e-7*eye(nDimxyz,nDimxyz);
end

%%%%%%
% EM %
%%%%%%
lambda_0_inv = zeros(nDimxyz,nDimxyz,numberOfComponents);
    
for niter = 1:iterEM,
    %%%%%%%%
    % E-step
    %%%%%%%%
    fprintf('EM iter %i/%i ',niter,iterEM);
    log_p_o=zeros(No,numberOfComponents);
    for m=1:numberOfComponents
        log_p_o(:,m) = gmmCalculateLikelihood(xyz, xyz_mu(:,m), inv(xyz_sigma(:,:,m)));
    end

    % up to a multiplicative constant
    p_o = exp(log_p_o-repmat(max(log_p_o,[],2),1,numberOfComponents));
    % for log-likelihood calculation
    p_o_nonorm = exp(log_p_o);
    
    % Calculate posterior probability (gamma)
    gamma_xyz_tmp = repmat(xyz_priors,[No 1]).*p_o;
    % normalize
    gamma_xyz = gamma_xyz_tmp ./ repmat(sum(gamma_xyz_tmp,2),[1 numberOfComponents]);
    
    % Small trick to avoid -Inf likelihood
    log_p_o_1 = sum(log(sum(repmat(xyz_priors,[No 1]).*p_o_nonorm,2)));
    
    
    % Calculate posteriors when data are patially observed (z is missing)
    log_p_o=zeros(N-No,numberOfComponents);
    for m=1:numberOfComponents
        % Calculate likelihood
        log_p_o(:,m) = gmmCalculateLikelihood(xy(:,No+1:N), xyz_mu(xy_ind,m), inv(xyz_sigma(xy_ind,xy_ind,m)));
    end
    % up to a multiplicative constant
    p_o = exp(log_p_o-repmat(max(log_p_o,[],2),1,numberOfComponents));
    
    % for likelihood calculation
    p_o_nonorm = exp(log_p_o);
    log_p_o_2 = sum(log(sum(repmat(xyz_priors,[N-No 1]).*p_o_nonorm,2)));
    log_p_o_total = log_p_o_1 + log_p_o_2;
    
    fprintf('Log-likelihood = %f\n',log_p_o_total);
    
    % Calculate posterior probability (gamma)
    gamma_xy_tmp = repmat(xyz_priors,[N-No 1]).*p_o;
    % normalize
    gamma_xy = gamma_xy_tmp ./ repmat(sum(gamma_xy_tmp,2),[1 numberOfComponents]);
    gamma_nm = [gamma_xyz;gamma_xy];
    
    % Estimate missing z
    z_full = zeros(nDimz,N,numberOfComponents);
    z_mse=zeros(nDimz,N-No,numberOfComponents);
    for m=1:numberOfComponents
        z_mse(:,:,m) = repmat(xyz_mu(z_ind,m),1,N-No) + (xyz_sigma(z_ind,xy_ind,m)/xyz_sigma(xy_ind,xy_ind,m))*(xy(:,No+1:N)-repmat(xyz_mu(xy_ind,m),1,N-No));
        z_full(:,:,m) = [z(:,:) z_mse(:,:,m)];
    end
    
    %%%%%%%%
    % M-step
    %%%%%%%%
    
    S = sum(gamma_nm);
    So = sum(gamma_nm(1:No,:));
    % update priors
    xyz_priors = S/N;
    for m=1:numberOfComponents
        o_prime_m = [x;y;z_full(:,:,m)];
        
        % update mean
        xyz_mu(:,m) = (1./S(m))* (o_prime_m*gamma_nm(:,m));
        
        % update covariance
        xyz_lambda_m = inv(xyz_sigma(:,:,m));
        lambda_0_inv(z_ind,z_ind,m) = inv(xyz_lambda_m(z_ind,z_ind));
        
        
        
        xyz_sigma(:,:,m)=(1./S(m))*((o_prime_m-repmat(xyz_mu(:,m),1,N)).*repmat(gamma_nm(:,m),1,nDimxyz)'*(o_prime_m-repmat(xyz_mu(:,m),1,N))' + (S(m)-So(m)).* lambda_0_inv(:,:,m));
        
        
        % regularize
        xyz_sigma(:,:,m)= xyz_sigma(:,:,m) + eye(nDimxyz)*1e-7;
        
    end
    
end

% Parameters
jgmmParam.xyz_priors = xyz_priors;
jgmmParam.xyz_mu = xyz_mu;
jgmmParam.xyz_sigma = xyz_sigma;
jgmmParam.M=numberOfComponents;
jgmmParam.nDimx = nDimx;
jgmmParam.nDimy = nDimy;
jgmmParam.nDimz = nDimz;
end

function par_j = gaussian_par_l2j(par_l)
%%% Auxiliar values
dX = numel(par_l.ind_x);
dY = numel(par_l.mu_y);
dZ = numel(par_l.ind_z);
% Check the existence of lin_zy
if ~isfield(par_l,'lin_zy')
    par_l.lin_zy = zeros(dZ,dY);
end
% Aux mean vector and precision matrix
mu_mod = zeros(dX+dY+dZ,1);
lambda = zeros(dX+dY+dZ,dX+dY+dZ);

%%% Parameters of Y
lambda(dX+(1:dY),dX+(1:dY)) = inv(par_l.cov_y) + ...
    par_l.lin_xy'*(par_l.cov_x\par_l.lin_xy) + ...
    par_l.lin_zy'*(par_l.cov_z\par_l.lin_zy);
mu_mod(dX+(1:dY)) = par_l.cov_y\par_l.mu_y - ...
    par_l.lin_xy'*(par_l.cov_x\par_l.ind_x) - ...
    par_l.lin_zy'*(par_l.cov_z\par_l.ind_z);

%%% Parameters of X|Y
lambda(1:dX,1:dX) = inv(par_l.cov_x) + ...
    par_l.lin_zx'*(par_l.cov_z\par_l.lin_zx);
lambda(1:dX,dX+(1:dY)) = -par_l.cov_x\par_l.lin_xy + ...
    par_l.lin_zx'*(par_l.cov_z\par_l.lin_zy);
lambda(dX+(1:dY),1:dX) = lambda(1:dX,dX+(1:dY))';
mu_mod(1:dX) = par_l.cov_x\par_l.ind_x - ...
    par_l.lin_zx'*(par_l.cov_z\par_l.ind_z);

%%% Parameters of Z|X,Y
lambda(dX+dY+(1:dZ),dX+dY+(1:dZ)) = inv(par_l.cov_z);
lambda(dX+dY+(1:dZ),1:dX) = -par_l.cov_z\par_l.lin_zx;
lambda(1:dX,dX+dY+(1:dZ)) = lambda(dX+dY+(1:dZ),1:dX)';
lambda(dX+dY+(1:dZ),dX+(1:dY)) = -par_l.cov_z\par_l.lin_zy;
lambda(dX+(1:dY),dX+dY+(1:dZ)) = lambda(dX+dY+(1:dZ),dX+(1:dY))';
mu_mod(dX+dY+(1:dZ)) = par_l.cov_z\par_l.ind_z;

%%% Preliminaries
% Invert the covariance matrix
par_j.cov = inv(lambda);
par_j.mu = par_j.cov*mu_mod;

end
