function [A,S] = MVCNMF(X,P,tol,maxiter,choice,Aini,Sini)
% A,S: output solution
% P: estimator number of endmembers
% tol: tolerance for a relative stopping condition
% maxiter: limit of iterations
% choice: 2 means using VCA to initialize the endmember and abundance matrix
tao = 0.1;
delta = 10;
%% PCA to reduce dimension:
band = size(X,1);      % number of bands
N = size(X,2);      % number of pixels
c = P;              % estimated number of endmembers
A = zeros(band,c);
if choice == 1
    for i=1:c
        index = ceil(rand()*N);
        A(:,i) = X(:,index);
    end
    S = zeros(c,N);     % initialize A and S for iteration
else
    A = Aini;
    S = Sini;
end
PrinComp = pca(X'); % number of row is sample number
meanData = mean(X,2);
err = 99999;
iterNum = 0;
%% project gradient descent using Armjio rules
while ( err>tol && iterNum<maxiter )
    [A,~] = steepdescent(X,S,A,tol,100,PrinComp(:,1:c-1),meanData,tao,1); 
    % to consider the sum-to-one constraint
    tX = [X; delta*ones(1,N)];
    tA = [A; delta*ones(1,c)];
    [S,~] = steepdescent(tX,tA,S,tol,100,PrinComp(:,1:c-1),meanData,tao,0);
    err = 0.5 * norm( (X(1:band,:) - A(1:band,:)*S), 2 )^2;
    iterNum = iterNum + 1;
    disp(iterNum);
end
end

    