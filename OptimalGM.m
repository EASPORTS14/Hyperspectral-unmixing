function [res] = OptimalGM(W,H,Hp,partial)
miu = 0.01;
alpha = 1;
Tmax = 5;
LC = 9e5;   % biger with the scale of matrix A & S£¨1e3,5,7£©
if partial=='H'
    Y = H;
    h = H;
    %LC = norm(W'*W,2);   
    for k=1:Tmax
        dF = W'*W*Y - W'*Hp + miu*ones(size(W',1),size(Y,2));
        h1 = Y - dF/LC;
        alp = (1+sqrt(1+4*alpha*alpha))/2;
        Y = h1 + (alpha-1)*(h1-h)/(alp);
        alpha = alp;
        h = h1;
    end
    res = h;
else
    Y = W;
    w = W;
    %LC = norm(H'*H,2);   % max eig of A'*A
    for k=1:Tmax
        dF = W*(H*H') - Hp*H';
        w1 = Y - dF/LC;
        alp = (1+sqrt(1+4*alpha*alpha))/2;
        Y = w1 + (alpha-1)*(w1-w)/(alp);
        alpha = alp;
        w = w1;
    end
    res = w;
end
end
    
        
     