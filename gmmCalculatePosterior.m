% Calculate beta=p(m|[x y])=p([x y]|m)p(m)
% Adapted from 2006 Sylvain Calinon, LASA Lab, EPFL, CH-1015 Lausanne, Switzerland, http://lasa.epfl.ch
% Thomas Hueber - CNRS/GIPSA-lab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function beta = gmmCalculatePosterior(data,priors, mu,sigma)

N = size(data,2);
M = length(priors);
nDim  = size(data,1);

% Calculate likelihood
log_likelihood = zeros(N,M);
sigma_inv = zeros(nDim,nDim,M);
for m=1:M
  sigma_inv(:,:,m)=inv(sigma(:,:,m));
  log_likelihood(:,m) = gmmCalculateLikelihood(data, mu(:,m), sigma_inv(:,:,m));
end
max_log_likelihood = max(log_likelihood,[],2);
likelihood = exp(log_likelihood-repmat(max_log_likelihood,1,M)); % up to a multiplicative constant

% Compute posterior probability (beta)
beta_tmp = repmat(priors,[N 1]).*likelihood;

% normalize !!
beta = beta_tmp ./ repmat(sum(beta_tmp,2),[1 M]);
alpha = 1-1e-5; % regularization parameter (posterior probability)
beta = alpha*beta+(1-alpha)/M; % avoid vanishing states

%% END of gmmCalculatePosterior FUNCTION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%