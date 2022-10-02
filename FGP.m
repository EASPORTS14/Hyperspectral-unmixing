function [X] = FGP(b,m,n,lamda,Tmax)
%% fast gradient project is used to handle total variation loss(unsmooth,discrete)
%Tmax = 20;
%lamda = 5e-6;
r = zeros(m-1,n);
p = zeros(m-1,n);
s = zeros(m,n-1);
q = zeros(m,n-1);
t1 = 1;
%% begin iteration
ite = 0;
while ite < Tmax
     ite = ite + 1;
     L = operatorL(r,s,m,n);
     [b1,b2] = operatorLT(b-lamda*L,m,n);
     [p1,q1] = operatorPP(r+b1/(8*lamda),s+b2/(8*lamda),m,n);
     t2 = (1+sqrt(1+4*t1*t1))/2;
     r = p1 + (t1-1)*(p1-p)/t2;
     s = q1 + (t1-1)*(q1-q)/t2;
     p = p1;  % update
     q = q1;
     t1 = t2;
end
X = b - lamda * operatorL(p,q,m,n);
end
%% inner utility function(operator)
function [L] = operatorL(p,q,m,n)
L = zeros(m,n);
L(1,1)=p(1,1)+q(1,1);
L(1,n)=p(1,n)-q(1,n-1);
L(m,1)=q(m,1)-p(m-1,1);
L(m,n)=-p(m-1,n)-q(m,n-1);
for i=2:m-1
    for j=2:n-1
        L(i,j)=p(i,j)+q(i,j)-p(i-1,j)-q(i,j-1);
    end
end      
for j=2:n-1
   L(1,j)=p(1,j)+q(1,j)-q(1,j-1);
   L(m,j)=q(m,j)-p(m-1,j)-q(m,j-1);
end
for i=2:m-1
   L(i,1)=p(i,1)+q(i,1)-p(i-1,1);
   L(i,n)=p(i,n)-p(i-1,n)-q(i,n-1);
end
end
function [p,q] = operatorLT(x,m,n)
p = zeros(m-1,n);
q = zeros(m,n-1);
for i=1:m-1
    for j=1:n
        p(i,j)=x(i,j)-x(i+1,j);
    end
end
for i=1:m
    for j=1:n-1
        q(i,j)=x(i,j)-x(i,j+1);
    end
end
end
function [r,s] = operatorPP(p,q,m,n)
r = zeros(m-1,n);
s = zeros(m,n-1);
     for i = 1:m-1
         for j = 1:n-1
             r(i,j)=p(i,j)/max(1,sqrt(p(i,j)^2+q(i,j)^2));
             s(i,j)=r(i,j);
         end
     end
     for i = 1:m-1
         r(i,n) = p(i,n)/max(1,abs(p(i,n)));
     end
     for j = 1:n-1
         s(m,j) = q(m,j)/max(1,abs(q(m,j))); 
     end
end

