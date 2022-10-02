function [S,grad,iter] = steepdescent(X,A,S,tol,maxiter,U,meanData,tao,cons)
% S, grad: output solution and its gradient
% iter: #iterations used
% X, A: constant matrices
% tol: stopping tolerance
% maxiter: limit of iterations
% U, meanData: principal components and data mean to calculate the volume
% tao: regularization parameter
% The constraint is only included when estimating A
if cons==1  % derive A
   % precalculation for volume constraint
   [~,c] = size(S);
   meanData = meanData * ones(1,c);
   C = [ones(1,c); zeros(c-1,c)];
   B = [zeros(1,c-1); eye(c-1)];
end
alpha = 1; 
beta = 0.1; 
sigma = 0.01;
for iter=1:maxiter
    % constraint on S^T
    if cons == 1
        Z = C + B*U'*(S-meanData);
        ZD = U * B' * pinv(Z);
        detz2 = det(Z)^2;
        f = sum(sum((X-S*A).^2)) + tao*det(Z)^2;
    end
    % gradient with respective to S
    if cons == 1
        grad = S*(A*A') - X*A' + tao*detz2*ZD;
    else
        grad = A'*A*S - A'*X;
    end
    projgrad = norm(grad(grad < 0 | S >0));
    if projgrad < tol
        break
    end
    % search step size 
    for inner_iter=1:50
        Sn = max(S - alpha*grad, 0); 
        d = Sn-S; 
        if cons == 1
            fn = sum(sum((X-Sn*A).^2)) + tao * det(C + B*U'*(Sn - meanData))^2;
            suff_decr = fn - f <= sigma*sum(sum(grad.*d));
        else       
            gradd=sum(sum(grad.*d)); 
            dQd = sum(sum((A'*A*d).*d));
            suff_decr = 0.99*gradd + 0.5*dQd < 0;
        end
        if inner_iter==1  % the first iteration determines whether we should increase or decrease alpha
            decr_alpha = ~suff_decr; 
            Sp = S;
        end
        if decr_alpha 
            if suff_decr
                S = Sn; break;
            else
                alpha = alpha * beta;
            end
        else
            if ~suff_decr | Sp == Sn
                S = Sp; break;
            else
                alpha = alpha/beta; 
                Sp = Sn;
            end
        end
    end
end