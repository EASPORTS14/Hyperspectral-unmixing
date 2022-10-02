function [A,S] = MDMD_NMF(X,Aini,Sini,Tmax,tolerance)
A = Aini;
S = Sini;
L = size(A,1);
J = size(A,2);
I = size(S,2);
delta = J;
lama = 0.01;
lams = 0.01*J;
mius = 8e-5;
miua = 1e-4;
augX = [X;delta*ones(1,I)];
itenum = 0;
%% begin iteration
while itenum<Tmax
      itenum = itenum + 1;
      disp(itenum);
      augA = [A;delta*ones(1,J)];
      ave = mean(mean(S));
      S = S - mius*(augA'*(augA*S-augX) - lams*(S - ave*ones(size(S))));
      A = A - miua*((A*S-X)*S' + lama*(A - ones(L,L)*A/L));
end
end