function [W,H] = L1_DNMF(X,Aini,Sini,Tmax)
delta = 20;
layer = 4;
miu = 0.001;
%% Pretraining Stage
Xorigin = X;
Wcell = cell(1,layer);
Hcell = cell(1,layer);
[~,P] = size(Aini);
ogm = 5;
for i = 1:layer
    disp(['pretrain stage handling layer ',num2str(i),'......']);
    if i==1
        Wcell{1} = Aini;
        Hcell{1} = Sini;
    else
        [Ae,~,~] = VCA(Hcell{i-1},'Endmembers', P,'verbose','on');
        Se = FCLS(Hcell{i-1}, Ae);
        Wcell{i} = Ae;
        Hcell{i} = Se;
    end
    itenum = 0;
    while itenum < ogm  % Stop criterion needs to be polished up!
        itenum = itenum + 1;
        disp(itenum);
        if i>1
            Hp = Hcell{i-1};
        else
            Hp = X;
        end
        Wcell{i} = OptimalGM(Wcell{i},Hcell{i},Hp,'W');
        Hcell{i} = OptimalGM(Wcell{i},Hcell{i},Hp,'H');
    end
end
%% Fine-tuning Stage
itenum = 0;
X = Xorigin;
augX = [X; delta*ones(1,size(X,2))];
while itenum < Tmax
    itenum = itenum + 1;
    fy = eye(size(Wcell{1},1),size(Wcell{1},1));   % equivalent to A1
    for i = 1:layer
        disp(['fine-tuning stage handling layer ',num2str(i),'......']);
        Wcell{i} = Wcell{i} .* (fy'*X*Hcell{i}') ./ (fy'*fy*Wcell{i}*Hcell{i}*Hcell{i}');
        fy = fy * Wcell{i};
        augfy = [fy; delta*ones(1,size(fy,2))];
        Hcell{i} = Hcell{i} .* (augfy'*augX) ./ (augfy'*augfy*Hcell{i}+miu*ones(size(Hcell{i})));
    end
end
W = Wcell{1};
H = Hcell{layer};
for i=2:layer
    W = W * Wcell{i};
end
end
% --------------------------------end of the Sparsity constrained L1_Deep NMF