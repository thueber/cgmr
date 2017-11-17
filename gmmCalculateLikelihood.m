%
% This function computes the Probability Density Function (PDF) of a
% multivariate Gaussian represented by means and covariance matrix.
%
% Inputs -----------------------------------------------------------------
%   o Data:  D x N array representing N datapoints of D dimensions.
%   o Mu:    D x K array representing the centers of the K GMM components.
%   o Sigma: D x D x K array representing the covariance matrices of the 
%            K GMM components.
% Outputs ----------------------------------------------------------------
%   o prob:  1 x N array representing the probabilities for the 
%            N datapoints.     
%
% ADAPTED FROM 2006 Sylvain Calinon, LASA Lab, EPFL, CH-1015 Lausanne,
%               Switzerland, http://lasa.epfl.ch
% Thomas Hueber - CNRS/GIPSA-lab

function log_prob = gmmCalculateLikelihood(Data, Mu, inv_Sigma)
  
  [nbVar,nbData] = size(Data);
if size(Mu,2)~=nbData
    Data = Data' - repmat(Mu',nbData,1);
else
    % Deal with the case when Mu is different for each frame (as in IC-GMR)
    Data = Data' - Mu';
end
log_prob = sum((Data*inv_Sigma).*Data, 2);
log_prob = -0.5*(log_prob - log(abs(det(inv_Sigma))+realmin)+nbVar*log(2*pi));