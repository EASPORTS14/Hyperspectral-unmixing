function [Ae,Se] = L2_1NMF(X,Aini,em,tolerance,Tmax)
% solving robust sparse unmixing(RSU) using optimizer ADMM
% inner function vect_soft is similiar to soft in the SSLRSU,
% so we can deduce without hesitation that such penalty items can be solved
% with ADMM and common derivative techniques!
E = Aini;
m = size(E,2);
A = zeros(m,size(X,2));
lamda = 0.3;
miu = 0.5;
P = zeros(size(X));
Q = A;
W = Q;
V1 = zeros(size(X));
V2 = zeros(size(A));
V3 = zeros(size(A));
%% begin iteration:
itenum = 0;
while itenum<Tmax
    itenum = itenum + 1;
    disp(itenum);
    cy = E*Q-X-V1;
    for i=1:size(P,1)
       [rowb] = vect_soft(cy(i,:),1/miu);
       P(i,:) = rowb; 
    end
    cy = Q-V2;
    for i=1:size(W,1)
       [rowb] = vect_soft(cy(i,:),lamda/miu);
       W(i,:) = rowb; 
    end
    A = Q - V3;
    A(A<0) = 0;
    Q = inv((E'*E+2*eye(m,m))) * (E'*(X+P+V1)+W+V2+A+V3);
    V1 = V1 - (E*Q-X-P);
    V2 = V2 - (Q-W);
    V3 = V3 - (Q-A);
end
%% post process: modify the number of endmember
ta = [];
for i=1:size(A,1)
    ta = [ta max(A(i,:))]; 
end
%ta = sum(A,2);
Ae = zeros(size(X,1),em);
Se = zeros(em,size(X,2));
for i=1:em
    [val,pos] = max(ta);
    disp([num2str(val),'  ',num2str(pos)]);
    Ae(:,i) = E(:,pos);
    Se(i,:) = A(pos,:);
    ta(1,pos) = -inf;
end
end
%% inner utility function
function [rowb] = vect_soft(b,t)
      m = max(norm(b,2),0);
      rowb = b * m / (m+t);
end