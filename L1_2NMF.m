%% Paper:Hyperspectral Unmixing Via L1_2 Sparsity-constrained Nonnegative Matrix Factorization
function [A, S, ARc, errRc, objRc] = L1_2NMF(X,AInit,tolObj,maxIter,flagS,SInit)
%% Input:
%     X: Hyperspectral data maxtrix (bandNum * sampleSize).
%     AInit: Endmember intial matrix (bandNum * emNum).
%     tolObj: Stop condition for difference of object function between two iteration.
%     maxIter: Maximun number of iterations.
%     flagS: determine whether to initialize S using prior knowledge
% Output:
%     A: resultant endmember matrix (bandNum * emNum).
%     S: resultant abundance matrix (emNum * sampleSize).
%     ARc: iteraive record for endmember matrix (emNum * iterNum * bandNum)
%     errRc: iterative record for error (iterNum).
%     objRc: iterative record for object value (iterNum).
%% Estimate the number of enNum endmembers using the HySime algorithm.
% Currenly, we omit this operation since we already know the number of endmember in synthetic data.
% Estimate the weight parameter fLamda according to the sparsity measure over X.
fDelta = 15;        % fDelta: Factor that control the strength of sum-to-one constraint.
emNum = size(AInit, 2);
bandNum = size(X, 1);
sampleNum = size(X, 2);
sqrtSampleNum = sqrt(sampleNum);
tmp = 0;
for l=1:bandNum
    xl =  X(l, :);
    if norm(xl,2) ~= 0
        tmp = tmp + ( sqrtSampleNum - (norm(xl,1)/norm(xl,2)) ) / (sqrtSampleNum -1);
    else
        tmp = tmp + ( sqrtSampleNum - 75 ) / (sqrtSampleNum -1);
    end
end
fLamda = tmp / sqrt(bandNum);
%% Record iteration.
errRc = zeros(1, maxIter);
objRc = zeros(1, maxIter);
ARecord = zeros(emNum, maxIter, bandNum);
% fLamda should be rescale to the level of spectral sample value
fLamda = fLamda / 500;
iterNum = 1;
A = AInit;
Xf = [ X; fDelta*ones(1, sampleNum) ];
Af = [ A; fDelta * ones(1, emNum) ];
if flagS ==0
    S = (Af'*Af)^(-1)*Af'*Xf;
else
    S = SInit;
end
err = 0.5 * norm( (Xf(1:bandNum,:) - Af(1:bandNum,:)*S), 2 )^2;
newObj = err + fLamda * fNorm(S, 1/2);
oldObj = 0;
disp(['flamda: ',num2str(fLamda)]);
disp(['err: ',num2str(err)]);
disp(['fnorm: ',num2str(fNorm(S, 1/2))]);
dispStr = ['Iteration ' num2str(iterNum),...
           ' loss = ' num2str(newObj)];
disp(dispStr);
errRc(iterNum) = err;
objRc(iterNum) = newObj;
%% Run iterations.
for i = 1:emNum
    ARecord(i, iterNum, :) = A(1:bandNum, i);
end
lowLimit = 0.0001;
while ( err >tolObj && (iterNum < maxIter) )
    oldObj = newObj;
    % update A
    check = Af*(S*S');   % added for the sake of W(which includes ...0000)
    check(find(check==0.0))  = lowLimit;
    A = Af .* (Xf*S') ./ check;
    % update S
    S(find(S<lowLimit))  = lowLimit;  % exist in the original version! It's a trick!
    S = S .* (A'*Xf) ./ (A'*A*S + 0.5 * fLamda * S.^(-1/2));
    Af = A;  % euqal to update Af matrix
    err = 0.5 * norm( (Xf(1:bandNum,:) - Af(1:bandNum,:)*S), 2 )^2;
    newObj = err + fLamda * fNorm(S, 1/2);
    iterNum = iterNum + 1;
    disp(['err: ',num2str(err)]);
    disp(['fnorm: ',num2str(fNorm(S, 1/2))]);
    dispStr = ['Iteration ' num2str(iterNum),': loss = ' num2str(newObj)];
    disp(dispStr);
    % record iteration.
    errRc(iterNum) = err;
    objRc(iterNum) = newObj;
    for i = 1:emNum
        ARecord(i, iterNum, :) = A(1:bandNum, i);
    end
end
A = A(1:bandNum, :);   % modify the dimension
ARc = ARecord(:, 1:iterNum, :);
errRc = errRc(1, 1:iterNum);
objRc = objRc(1, 1:iterNum);
end