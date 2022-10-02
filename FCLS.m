function [out] = FCLS(HIM,M)
% Fully Constrained Linear Spectral Unmixing
% Perform a Linear least squares with nonnegativity constraints.
% Input:   HIM : hyperspectral image cube [nbands x nsamples]
%          M   : set of p endmembers [nbands x p].
% Output:  out : fractions [p x nsamples] 
[~,pixel] = size(HIM);
[band,p] = size(M);
Delta = 1e-5; % should be an small value

N = zeros(band+1,p);
N(1:band,1:p) = Delta*M;
N(band+1,:) = ones(1,p);
s = zeros(band+1,1);
out = zeros(pixel,p);
for i = 1:pixel
    s(1:band) = Delta*HIM(:,i);
    s(band+1) = 1;
    Abundances = lsqnonneg(N,s);
    out(i,:) = Abundances;
end
out = out';
end