function [A,S] = GLNMF(X,Aini,Sini,Tmax)
alpha = -1;    % types: four choices: 0,2,-inf,otherwise
c = 2;
A = Aini;
S = Sini;
itenum = 0;
band = size(X,1);
W = zeros(band,band);
while itenum < Tmax
    error = X - A*S;
    list_error = zeros(band,1);
    for i = 1:band
        list_error(i,1) = norm(error(i,:),2)^2;
    end
    scalar = max(list_error) / 10.0;
    for i = 1:band
        e = list_error(i,1) / scalar;
        [wloss] = general_loss(e,alpha,c);
        W(i,i) = wloss;
    end
    disp(diag(W));
    Xe = W.^0.5 * X;
    Ae = W.^0.5 * A;
    [Ae,S] = L1_2NMF(Xe,Ae,1e-6,20,1,S);
    for i=1:band
        if W(i,i) ~= 0
            W(i,i) = W(i,i)^(-0.5);
        end
    end
    A = W * Ae;
    itenum = itenum + 1;
end
end
%% ei-loss function : NAN problems need to be solved
function [wloss] = general_loss(e,alpha,c)
     if alpha == 2
         wloss = 1.0/(c*c);
     else
         if alpha == 0
             wloss = 2.0/(e^2+2*c*c);
         else
             if alpha == -inf
                 wloss = c^(-2) * exp(-0.5*e^2/(c^2));
             else
                 wloss = c^(-2) * ((1+((e/c)^2)/(abs(2-alpha)))^(0.5*alpha-1));
             end
         end
     end       
end      