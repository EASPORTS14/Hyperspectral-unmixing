function [A,S] = SDNMF(X,Aini,Sini,Tmax)
delta = 15;
layer = 3;
%% Pretraining Stage
Xorigin = X;
Acell = cell(1,layer);
Scell = cell(1,layer);
[L,P] = size(Aini);
for i = 1:layer
    disp(['pretrain stage handling layer ',num2str(i),'......']);
    if i == 1
        Acell{1} = Aini;
        Scell{1} = Sini;
    else
        [Ae,~,~] = VCA(Scell{i-1},'Endmembers', P,'verbose','on');
        Se = FCLS(Scell{i-1}, Ae);
        Acell{i} = Ae;
        Scell{i} = Se;
    end
    itenum = 0;
    while itenum < Tmax  % Stop criterion needs to be polished up!
        itenum = itenum + 1;
        if i>1
            X = Scell{i-1};
        end
        augX = [X; delta*ones(1,size(X,2))];
        Acell{i} = Acell{i} .* (X*Scell{i}') ./ (Acell{i}*(Scell{i}*Scell{i}'));
        augA = [Acell{i}; delta*ones(1,size(Acell{i},2))];
        Scell{i} = Scell{i} .* (augA'*augX) ./ (augA'*augA*Scell{i});
    end
end
%% Fine-tuning Stage
itenum = 0;
X = Xorigin;
augX = [X; delta*ones(1,size(X,2))];
while itenum < Tmax
    itenum = itenum + 1;
    fy = eye(L,L);   % equivalent to A1
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
        Scell{i} = Scell{i} .* (augfy'*augX) ./ (augfy'*augfy*Scell{i});
    end
end
A = Acell{1};
S = Scell{layer};
for i = 2:layer
    A = A * Acell{i};
end
end
% --------------------------------end of the Sparsity constrained deep NMF