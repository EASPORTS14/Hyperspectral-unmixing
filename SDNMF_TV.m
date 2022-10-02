function [A,S] = SDNMF_TV(X,Aini,Sini,Tmax)
lamda = 0.2;
alpha = 0.005;
miu = 1e3;
delta = 15;
layer = 3;
%% Pretraining Stage
Xorigin = X;
Acell = cell(1,layer);
Scell = cell(1,layer);
Lcell = cell(1,layer);
N = size(Sini,2);
[band,P] = size(Aini);
m = sqrt(N);
for i = 1:layer
    disp(['pretrain stage handling layer ',num2str(i),'......']);
    if i == 1
        Acell{1} = Aini;
        Scell{1} = Sini;
        Lcell{1} = zeros(P,N);
    else
        [Ae,~,~] = VCA(Scell{i-1},'Endmembers', P,'verbose','on');  % P means the number of endmember!
        Se = FCLS(Scell{i-1}, Ae);
        Acell{i} = Ae;
        Scell{i} = Se;
        Lcell{i} = Se;
    end
    itenum = 0;
    while itenum < Tmax  % Stop criterion needs to be polished up!
        itenum = itenum + 1;
        if i > 1
            X = Scell{i-1};
        end
        augX = [X; delta*ones(1,size(X,2))];
        Acell{i} = Acell{i} .* (X*Scell{i}') ./ (Acell{i}*(Scell{i}*Scell{i}'));
        augA = [Acell{i}; delta*ones(1,size(Acell{i},2))];
        Scell{i}(Scell{i}<eps) = eps;
        Stemp = zeros(size(Scell{i}));
        S1 = Scell{i} .* (augA'*augX+miu*Lcell{i}) ./ (augA'*augA*Scell{i} + 0.5*lamda*(Scell{i}.^(-0.5)) + miu*Scell{i});
        S2 = Scell{i} .* (augA'*augX+miu*Lcell{i}) ./ (augA'*augA*Scell{i} + miu*Scell{i});
        Stemp(Scell{i}>=10^-4) = S1(Scell{i}>=10^-4);
        Stemp(Scell{i}<10^-4) = S2(Scell{i}<10^-4);
        Scell{i} = Stemp;
        for row  = 1:P
           ss = reshape(Scell{i}(row,:),m,m);
           tt = FGP(ss,m,m,5e-6,20);
           Lcell{i}(row,:) = reshape(tt,1,N);
        end
        disp(itenum);
    end
end
%% Fine-tuning Stage
itenum = 0;
X = Xorigin;
augX = [X; delta*ones(1,size(X,2))];
while itenum < Tmax
    itenum = itenum + 1;
    fy = eye(band,band);   % equivalent to A1
    for i = 1:layer
        disp(['fine-tuning stage handling layer ',num2str(i),'......']);
        if i == layer
            eS = Scell{layer};
        else
            eS = Acell{i+1};
            for j = (i+2):layer
                eS = eS * Acell{j};
            end
            eS = eS * Scell{layer};
        end
        Acell{i} = Acell{i} .* (fy'*X*eS') ./ (fy'*fy*Acell{i}*(eS*eS'));
        fy = fy * Acell{i};
        augfy = [fy; delta*ones(1,size(fy,2))];
        Scell{i}(Scell{i}<eps) = eps;
        Stemp = zeros(size(Scell{i}));
        S1 = Scell{i} .* (augfy'*augX + miu*Lcell{i}) ./ (augfy'*augfy*Scell{i}+0.5*lamda*Scell{i}.^(-0.5)+miu*Scell{i});
        S2 = Scell{i} .* (augfy'*augX + miu*Lcell{i}) ./ (augfy'*augfy*Scell{i}+miu*Scell{i});
        Stemp(Scell{i}>=10^-4) = S1(Scell{i}>=10^-4);
        Stemp(Scell{i}<10^-4) = S2(Scell{i}<10^-4);
        Scell{i} = Stemp;
        for row = 1:P
           ss = reshape(Scell{i}(row,:),m,m);
           tt = FGP(ss,m,m,5e-6,20);
           Lcell{i}(row,:) = reshape(tt,1,N);
        end
    end
end
A = Acell{1};
S = Scell{layer};
for i = 2:layer
    A = A * Acell{i};
end
end
% --------------------------------end of the Sparsity constrained NMF_TV