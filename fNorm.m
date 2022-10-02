%% Matrix fractional norm implementation 
function f = fNorm(X, Frac)
    elemFrac = X.^Frac;
    upperLimit = 100;
    if Frac < 0
        elemFrac(find(elemFrac>upperLimit)) = upperLimit;  % return the index of elements
    end
    f = sum(sum(elemFrac));
end