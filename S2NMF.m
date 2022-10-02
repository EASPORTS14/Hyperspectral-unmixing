function [M,A] = S2NMF(Y,Aini,Sini,tolerance,Tmax)
miu = 0;
lam1 = 0.11;
lam2 = 0.1;
delta = 15;
n = size(Y,2);
kernel = 1e6;
M = Aini;
A = Sini;
WL = get_matrice(Y,ceil(sqrt(n)),kernel);
WG = zeros(size(WL));
lowLimit = 1e-6;
%% begin iteration
itn = 1;
H = eye(size(WL)) - miu*WG - (1-miu)*WL;
augY = [Y; delta*ones(1,size(Y,2))];
while itn < Tmax
    disp(itn);
    M = M .* (Y*A') ./ (M*(A*A'));
    augM = [M; delta*ones(1,size(M,2))];
    A(A<lowLimit) = lowLimit;
    A = A .* (augM'*augY) ./ (augM'*augM*A + 0.25*lam1*A.^(-0.5) + lam2*A*(H*H'));
    itn = itn + 1;
end
end
%% inner utility function
function [WL] = get_matrice(X,segments,kernel)
[label,~,~,~] = superPixel(X,segments,1);  % we first handle the cluster problem on the original image.'1' means using primary features
n = size(label,2);
WL = zeros(n,n);
for i = 1:n
    for j =1:n
        if label(1,i) == label(1,j)
            v1 = X(:,i);
            v2 = X(:,j);
            WL(i,j) = exp(-1*(norm((v1-v2),2)^2)/kernel);
            WL(j,i) = WL(i,j);
            %disp(WL(i,j));
        else
            WL(i,j) = 0;
            WL(j,i) = 0;
        end
    end
end
for row = 1:n
   rs = sum(WL(row,:));
   WL(row,:) = WL(row,:)/rs;
end
end