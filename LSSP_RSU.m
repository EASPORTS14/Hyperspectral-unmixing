function [A,X] = LSSP_RSU(Y,Aini,Sini,tolerance,Tmax)
% solving robust sparse unmixing(LSSP_RSU) using optimizer ADMM.
% inner function vect_soft is similiar to soft in the SSLRSU,
% so we can deduce without hesitation that such penalty items can be solved
% with ADMM and common derivative techniques!
%% Important initialization
A = Aini;
U = Sini;
m = size(A,2);
n = size(Sini,2);  % number of pixels
lamda = 0.1;
miu = 0.01;
D1 = zeros(size(Y));
D2 = zeros(size(U));
D3 = zeros(size(U));
D4 = zeros(size(U));
%% Computer operator matrix M,Q
shift = zeros(n,n);
for i=1:(n-1)
    shift(i,i+1)=1; 
end
M = eye(n,n)-shift;
Q = zeros(n,n);
yh = Y*M;                % !!ensure the dimension of M AND Q !!
for i=1:n
    Q(i,i)=1.0/(norm(yh(:,i),2)); 
end
%% begin iteration:
itenumk = 0;
while itenumk<Tmax
    itenumk = itenumk + 1;
    disp(itenumk);
    [P] = vect_soft(A*U-Y-D1,1.0/miu);
    [W] = vect_soft(U-D2,lamda/miu);
    [R] = soft(U*M*Q-D3,2*lamda/miu);
    X = max(U-D4,0);
    a1 = Y+P+D1;
    a2 = W+D2;
    a3 = R*(M*Q)' + D3*M*Q;
    a4 = X+D4;
    mm = M(2:m+1,2:m+1);
    qq = Q(2:m+1,2:m+1);
    U = inv(A'*A+((mm*qq)*(mm*qq)')+2*eye(m,m)) * (A'*a1+a2+a3+a4); 
    D1 = D1 - (A*U-Y-P);
    D2 = D2 - (U-W);
    D3 = D3 - (U*M*Q-R);
    D4 = D4 - (U-X);
end
end

%% inner utility function
function [sm] = soft(y,t)
     [h,w] = size(y);
     sm = zeros(h,w);
     for i=1:h
         for j=1:w
             sm(i,j) = sign(y(i,j)) * max(abs(y(i,j)-t),0);
         end
     end
end
%% inner utility function
function [vm] = vect_soft(b,t)
      m = max(norm(b,2),0);
      vm = b * m / (m+t);
end