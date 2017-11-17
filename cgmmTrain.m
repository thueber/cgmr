%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training routine for the following graphical model:
%
%    m
%  / | \
% y->x->z
%
% with m is unobserved and z is partially observed
% and p(x,y) & p(z,x) are modeled using GMM
% pdf for p(x,y) can be used for initialization
% Thomas Hueber - Laurent Girin - Xavi Alameda - GIPSA-lab/CNRS/INRIA - 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cgmmParam = cgmmTrain(x,y,z,xy_priors,xy_mu,xy_sigma,nb_iter_em)
%dbstop if error
% Set some initial variables
M = size(xy_mu,2);
No = size(z,2);
N = size(x,2);
fprintf('No=%i,N=%i,M=%i\n',No,N,M);

nDimx = size(x,1);
nDimy = size(y,1);
nDimz = size(z,1);
nDimxy = nDimx+nDimy;
%nDimxz = nDimx+nDimz;
%nDimyz = nDimy+nDimz;
nDimxyz = nDimx+nDimy+nDimz;

%xy_ind = [1:(nDimxy)];
%z_ind = [nDimxy+1:nDimxyz];
x_ind = [1:(nDimx)];
y_ind = [nDimx+1:nDimxy];
%zx_ind = [z_ind x_ind];
%xz_ind = [x_ind z_ind];
%zy_ind = [z_ind y_ind];
%yz_ind = [y_ind z_ind];

xy = [x;y];
%xyz = [x(:,1:No);y(:,1:No);z];
%xz = [x(:,1:No);z];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize p(x,y,z) by the suboptimal procedure %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate beta=p(m|[x y])=p([x y]|m)p(m)
beta = gmmCalculatePosterior(xy,xy_priors,xy_mu,xy_sigma);


xyz_priors = xy_priors;
xyz_mu = zeros(nDimxyz,M);
xyz_sigma = zeros(nDimxyz,nDimxyz,M);

reg_U = eye(nDimx)*1e-7;
reg_V = eye(nDimz)*1e-7;

% Init e,R,A,b,C,d,U,V (using Xavi procedure)
%%%%%%%%%%%%
 % Coherent soft (EM-like)
 fprintf('Initialization coherent (soft) of IC-GMM\n');
 % Declare variables
 S = sum(beta); So = sum(beta(1:No,:));
 S_x = x*beta;
 S_y = y*beta;
 % Declare zero-valued variables
 S_z = zeros(nDimz,M);
 S_xx = zeros(nDimx,nDimx,M);
 S_zx = zeros(nDimz,nDimx,M);
 S_xy = zeros(nDimx,nDimy,M);
 S_yy = zeros(nDimy,nDimy,M);
 
 % Parameters
 % y
 e = S_y./repmat(S,nDimy,1);
 R = zeros(nDimy,nDimy,M);
 % x|y
 A = zeros(nDimx,nDimy,M);
 b = zeros(nDimx,M);
 U = zeros(nDimx,nDimx,M);
 % z|x
 C = zeros(nDimz,nDimx,M);
 d = zeros(nDimz,M);
 V = zeros(nDimz,nDimz,M);
 
 for m = 1:M
    % This code is very similar to the M-step
    % (at least to the M-Z|X step)
    S_z(:,m) = z*beta(1:No,m);
    S_xx(:,:,m) = x.*repmat(beta(:,m),1,nDimx)'*x';
    S_zx(:,:,m) = z(:,1:No).*repmat(beta(1:No,m),1,nDimz)'*x(:,1:No)';
    S_xy(:,:,m) = x.*repmat(beta(:,m),1,nDimx)'*y';
    S_yy(:,:,m) = y.*repmat(beta(:,m),1,nDimy)'*y';
    
    % Let's now for C, d and V
    R(:,:,m) = (1./S(m)).*(y-repmat(e(:,m),1,N)).*repmat(beta(:,m),1,nDimy)'*(y-repmat(e(:,m),1,N))';
    A(:,:,m) = (S_xy(:,:,m)-(1./S(m)).*S_x(:,m)*S_y(:,m)')/(S_yy(:,:,m)-(1./S(m)).*S_y(:,m)*S_y(:,m)');
    b(:,m) = (1./S(m)).*(S_x(:,m)-A(:,:,m)*S_y(:,m));
    U(:,:,m) = (1./S(m)).*(x-A(:,:,m)*y-repmat(b(:,m),1,N)).*repmat(beta(:,m),1,nDimx)'*((x-A(:,:,m)*y-repmat(b(:,m),1,N)))'+reg_U;
    %C(:,:,m) = (S(m)/So(m))*(S_zx(:,:,m)-(1./S(m)).*S_z(:,m)*S_x(:,m)')/(S_xx(:,:,m)-(1./S(m)).*S_x(:,m)*S_x(:,m)');
    %d(:,m) = (1./S(m)).*(S_z(:,m)*S(m)/So(m)-C(:,:,m)*S_x(:,m));
    %V(:,:,m) = (1./So(m)).*( (z(:,1:No)-C(:,:,m)*x(:,1:No)-repmat(d(:,m),1,No)).*repmat(beta(1:No,m),1,nDimz)'*(z(:,1:No)-C(:,:,m)*x(:,1:No)-repmat(d(:,m),1,No))' )+reg_V;
    
    C(:,:,m) = eye(nDimz,nDimx);
    d(:,m) = zeros(nDimz,1);
    V(:,:,m) =  U(:,:,m) + A(:,:,m)*R(:,:,m)*A(:,:,m)' + reg_V;
 end

%%%%%%
% EM %
%%%%%%
% niter = 1;
% delta_loglikelihood = Inf;
% last_log_p_o_total = 0;

%while abs(delta_loglikelihood) > nb_iter_em
for niter = 1:nb_iter_em
    fprintf('EM iter %i: ',niter);
    
    % E-step
    %%%%%%%%%
    % Calculate posteriors when data are fully observed
    % Calculate likelihood
    log_p_y=zeros(No,M);
    log_p_x_y=zeros(No,M);
    log_p_z_x=zeros(No,M);
    log_p_o=zeros(No,M);
    p_o=zeros(No,M);
    gamma_xyz_tmp=zeros(No,M);
    gamma_xyz=zeros(No,M);
    for m=1:M
        log_p_y(:,m) = gmmCalculateLikelihood(y(:,1:No), e(:,m), inv(R(:,:,m)));
        log_p_x_y(:,m) = gmmCalculateLikelihood(x(:,1:No), A(:,:,m)*y(:,1:No)+repmat(b(:,m),1,No), inv(U(:,:,m)));
        log_p_z_x(:,m) = gmmCalculateLikelihood(z(:,1:No), C(:,:,m)*x(:,1:No)+repmat(d(:,m),1,No), inv(V(:,:,m)));
    end
    log_p_o = log_p_y+log_p_x_y+log_p_z_x;
    p_o = exp(log_p_o-repmat(max(log_p_o,[],2),1,M)); % up to a multiplicative constant
    
    p_o_nonorm = exp(log_p_o); % for log-likelihood calculation
    
    % Calculate posterior probability (gamma)
    gamma_xyz_tmp = repmat(xyz_priors,[No 1]).*p_o;
    gamma_xyz = gamma_xyz_tmp ./ repmat(sum(gamma_xyz_tmp,2),[1 M]);%normalize !
    
    log_p_o_1 = sum(log(sum(repmat(xyz_priors,[No 1]).*p_o_nonorm,2)));
        
    % Calculate posteriors when data are patially observed (z is missing)
    % Calculate likelihood
    log_p_y=zeros(N-No,M);
    log_p_x_y=zeros(N-No,M);
    log_p_o=zeros(N-No,M);
    p_o=zeros(N-No,M);
    gamma_xy_tmp=zeros(N-No,M);
    gamma_xy=zeros(N-No,M);
    for m=1:M
        log_p_y(:,m) = gmmCalculateLikelihood(y(:,No+1:N), e(:,m), inv(R(:,:,m)));
        log_p_x_y(:,m) = gmmCalculateLikelihood(x(:,No+1:N), A(:,:,m)*y(:,No+1:N)+repmat(b(:,m),1,N-No), inv(U(:,:,m)));
    end
    log_p_o = log_p_y+log_p_x_y;
    p_o = exp(log_p_o-repmat(max(log_p_o,[],2),1,M)); % up to a multiplicative constant
    
    p_o_nonorm = exp(log_p_o); % for likelihood calculation
    log_p_o_2 = sum(log(sum(repmat(xyz_priors,[N-No 1]).*p_o_nonorm,2)));
    log_p_o_total = log_p_o_1 + log_p_o_2;
    fprintf('Log-likelihood = %f\n',log_p_o_total);
    
%     delta_loglikelihood = (last_log_p_o_total - log_p_o_total);
%     last_log_p_o_total = log_p_o_total;
%     niter = niter+1;
%     
%     fprintf('Log-likelihood = %f (improvment %.3f)\n',log_p_o_total,delta_loglikelihood);
    
    % Calculate posterior probability (gamma)
    gamma_xy_tmp = repmat(xyz_priors,[N-No 1]).*p_o;
    gamma_xy = gamma_xy_tmp ./ repmat(sum(gamma_xy_tmp,2),[1 M]);%normalize !
    
    gamma_nm = [gamma_xyz;gamma_xy];
    
    % Estimate missing z
    z_full = zeros(nDimz,N,M);
    z_mse=zeros(nDimz,N-No,M);
    for m=1:M
        z_mse(:,:,m) = C(:,:,m)*x(:,No+1:N)+repmat(d(:,m),1,N-No);
        z_full(:,:,m) = [z(:,:) z_mse(:,:,m)];
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Debug - reconstruct missing data
    %z2 = zeros(nDimz,N-No);
    %for m=1:M
    %    z2 = z2 +z_mse(:,:,m).*repmat(gamma_xy(:,m),1,nDimz)';
    %end
    %z3 = [z z2];
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % M-step
    %%%%%%%%
    S_z = zeros(nDimz,M);
    S_xy = zeros(nDimx,nDimy,M);
    S_xx = zeros(nDimx,nDimx,M);
    S_yy = zeros(nDimy,nDimy,M);
    S_zx = zeros(nDimz,nDimx,M);
    S = sum(gamma_nm);
    S_x = x*gamma_nm;
    S_y = y*gamma_nm;
    for m=1:M
        S_z(:,m) = z_full(:,:,m)*gamma_nm(:,m);
        S_xy(:,:,m) = x.*repmat(gamma_nm(:,m),1,nDimx)'*y';
        S_xx(:,:,m) = x.*repmat(gamma_nm(:,m),1,nDimx)'*x';
        S_yy(:,:,m) = y.*repmat(gamma_nm(:,m),1,nDimy)'*y';
        S_zx(:,:,m) = z_full(:,:,m).*repmat(gamma_nm(:,m),1,nDimz)'*x';
    end
    
    xyz_priors = S/N;
    e = repmat(1./S,nDimy,1).*(y*gamma_nm);
    
    for m=1:M
        R(:,:,m) = (1./S(m)).*(y-repmat(e(:,m),1,N)).*repmat(gamma_nm(:,m),1,nDimy)'*(y-repmat(e(:,m),1,N))';
        A(:,:,m) = (S_xy(:,:,m)-(1./S(m)).*S_x(:,m)*S_y(:,m)')/(S_yy(:,:,m)-(1./S(m)).*S_y(:,m)*S_y(:,m)');
        b(:,m) = (1./S(m)).*(S_x(:,m)-A(:,:,m)*S_y(:,m));
        C(:,:,m) = (S_zx(:,:,m)-(1./S(m)).*S_z(:,m)*S_x(:,m)')/(S_xx(:,:,m)-(1./S(m)).*S_x(:,m)*S_x(:,m)');
        d(:,m) = (1./S(m)).*(S_z(:,m)-C(:,:,m)*S_x(:,m));
        U(:,:,m) = (1./S(m)).*(x-A(:,:,m)*y-repmat(b(:,m),1,N)).*repmat(gamma_nm(:,m),1,nDimx)'*((x-A(:,:,m)*y-repmat(b(:,m),1,N)))'+reg_U;
        V(:,:,m) = (1./S(m)).*((z_full(:,:,m)-C(:,:,m)*x-repmat(d(:,m),1,N)).*repmat(gamma_nm(:,m),1,nDimz)'*((z_full(:,:,m)-C(:,:,m)*x-repmat(d(:,m),1,N)))'+(sum(gamma_nm(No+1:N,m)).*V(:,:,m)))+reg_V;
    end
end % iter EM


% Calculate all the variables needed for inference
e_star = zeros(nDimx,M);
R_star = zeros(nDimx,nDimx,M);
A_star = zeros(nDimy,nDimx,M);
b_star = zeros(nDimy,M);
U_star = zeros(nDimy,nDimy,M);
C_star = zeros(nDimz,nDimx,M);
d_star = zeros(nDimz,M);
V_star = zeros(nDimz,nDimz,M);

f = zeros(nDimz,M);
P = zeros(nDimz,nDimz,M);

for m=1:M
    U_star(:,:,m)=inv(inv(R(:,:,m))+A(:,:,m)'/(U(:,:,m))*A(:,:,m));
    A_star(:,:,m)= (U_star(:,:,m)*A(:,:,m)'/(U(:,:,m)));
    b_star(:,m) = U_star(:,:,m)*((R(:,:,m))\e(:,m)-A(:,:,m)'/(U(:,:,m))*b(:,m));
    
    R_star(:,:,m) = U(:,:,m) + A(:,:,m)*R(:,:,m)*A(:,:,m)';
    e_star(:,m) = A(:,:,m)*e(:,m)+b(:,m);
    
    
    V_star(:,:,m) = inv(inv(R_star(:,:,m))+C(:,:,m)'/(V(:,:,m))*C(:,:,m));
    C_star(:,:,m) = V_star(:,:,m)*C(:,:,m)'/(V(:,:,m));
    
    d_star(:,m) = V_star(:,:,m)*((R_star(:,:,m))\e_star(:,m)-C(:,:,m)'/(V(:,:,m))*d(:,m));
    
    f(:,m) = C(:,:,m)*e_star(:,m) + d(:,m);
    P(:,:,m) = V(:,:,m) + C(:,:,m)*R_star(:,:,m)*C(:,:,m)';
end

% Store parameters needed for inference (used by cgmmMap function)
cgmmParam.U_star = U_star;
cgmmParam.A_star = A_star;
cgmmParam.b_star = b_star;
cgmmParam.e_star = e_star;
cgmmParam.R_star = R_star;
cgmmParam.V_star = V_star;
cgmmParam.C_star = C_star;
cgmmParam.d_star = d_star;

cgmmParam.U = U;
cgmmParam.A = A;
cgmmParam.b = b;
cgmmParam.e = e;
cgmmParam.R = R;
cgmmParam.V = V;
cgmmParam.C = C;
cgmmParam.d = d;

cgmmParam.f = f;
cgmmParam.P = P;

cgmmParam.xyz_priors = xyz_priors;
cgmmParam.xyz_mu = xyz_mu;
cgmmParam.xyz_sigma = xyz_sigma;

cgmmParam.M = M;
cgmmParam.nDimx = nDimx;
cgmmParam.nDimy = nDimy;
cgmmParam.nDimz = nDimz;
cgmmParam.x_ind = x_ind;
cgmmParam.y_ind = y_ind;

%% END of FUNCTION cgmmTrain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
