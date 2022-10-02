function [Ae,Se] = SSLRSU(Y,Aini,em,tolerance,Tmax)
% solving robust sparse unmixing(RSU) using optimizer ADMM.
% inner function vect_soft is similiar to soft in the SSLRSU,
% so we can deduce without hesitation that such penalty items can be solved
% with ADMM and common derivative techniques!
A = Aini;
n = size(A,2);
lamda = 3e-3;
miu = 0.5;
tao = 1;
sigma = 0.1;
U = inv(A'*A+3*eye(n,n)) * A' * Y;
V1 = A*U;
V2 = U;
V3 = U;
V4 = U;
D1 = zeros(size(Y));
D2 = zeros(size(U));
D3 = zeros(size(U));
D4 = zeros(size(U));
H1 = zeros(n,n);
H2 = zeros(size(U));
%% begin iteration:
itenumk = 0;
while itenumk<Tmax
    itenumk = itenumk + 1;
    disp(itenumk);
    % calculate H1 AND H2:
    temp = U - D2;
    for i=1:n
       H1(i,i) = 1.0/(norm(temp(i,:),2)+sigma); 
    end
    H2 = (temp+sigma).^(-1);
    % inner loop for 5 times:
    for i=1:2
        U = inv(A'*A+3*eye(n,n)) * (A'*(V1+D1)+V2+D2+V3+D3+V4+D4);
        V1 = (Y+miu*(A*U-D1))/(1+miu);
        [sm] = soft(U-D2,(lamda/miu)*(H1*H2));
        V2 = sm;
        [~,sv,~] = svd(V3);
        b = diag(sv);         % sigular value decompose and record the diag vector
        [u,sv,v] = svd(U-D3);
        drank = min(size(sv,1),size(sv,2));
        for j=1:drank
            sv(j,j) = max(sv(j,j)-tao*b(j,:)/miu); 
        end
        V3 = u*sv*v';
        V4 = max(U-D4,0);
        D1 = D1-A*U+V1;
        D2 = D2-U+V2;
        D3 = D3-U+V3;
        D4 = D4-U+V4;
    end
end
%% post process: modify the number of endmember
% ta = [];
% for i=1:size(U,1)
%     ta = [ta max(U(i,:))]; 
% end
ta = sum(U,2);  % sum of each row
Ae = zeros(size(Y,1),em);
Se = zeros(em,size(Y,2));
for i=1:em
    [val,pos] = max(ta);
    disp([num2str(val),'  ',num2str(pos)]);
    Ae(:,i) = A(:,pos);
    Se(i,:) = U(pos,:);
    ta(pos,1) = -inf;
end
end
%% inner utility function
function [sm] = soft(y,t)
     [h,w] = size(y);
     sm = zeros(h,w);
     for i=1:h
         for j=1:w
             sm(i,j) = sign(y(i,j))*max(abs(y(i,j)-t(i,j)),0);
         end
     end
end