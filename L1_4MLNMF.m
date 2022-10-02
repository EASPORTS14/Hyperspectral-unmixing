function [A,S] = L1_4MLNMF(X,Aini,Sini,tolerance,Tmax,option)
% ---option == 0 means using VCA-FCLS to initialize the layer
alpha = 0.001;
tao = 25;
layer = 10;
lowLimit = 1e-7;
P = size(Aini,2);
Acell = cell(1,layer);
Scell = cell(1,layer);
%% Handle every layer without considering the global loss(X-AS):
for i = 1:layer
    disp(['pretrain stage handling layer ',num2str(i),'......']);
    if i == 1
        Acell{i} = Aini;
        Scell{i} = Sini;
    else
        if option == 0
            [Ae,~,~] = VCA(Scell{i-1},'Endmembers', P,'verbose','on');
            Se = FCLS(Scell{i-1}, Ae);
            Acell{i} = Ae;
            Scell{i} = Se;
        else
            Acell{i} = rand(P,P);
            Scell{i} = Scell{i-1};
        end
        X = Scell{i-1};
    end
    itn = 1;
    Ap = Acell{i};
    count = 0;
    while itn < Tmax
        disp(itn);
        aa = alpha*exp(-itn/tao);
        as = 2*aa;
        Acell{i}(Acell{i}<lowLimit) = lowLimit;
        Acell{i} = Acell{i} .* (X*Scell{i}') ./ (Acell{i}*Scell{i}*Scell{i}' + 0.5*aa*(Acell{i}.^(-0.75)));
        Acell{i} = real(Acell{i});
        Scell{i}(Scell{i}<lowLimit) = lowLimit;
        Scell{i} = Scell{i} .* (Acell{i}'*X) ./ (Acell{i}'*Acell{i}*Scell{i} + 0.5*as*(Scell{i}.^(-0.5)));
        Scell{i} = real(Scell{i});
        if itn ~= 1
            err = norm(Acell{i}-Ap);
            disp([num2str(err) ,' @ ', num2str(itn)])
            if(err<tolerance)
                count = count + 1;
                if (count == 10)
                    disp(['done!' 'Tolerance:' num2str(err) '@' num2str(itn)]);
                    break;
                end
            else
                count = 0;
            end
        end
        Ap = Acell{i};
    
        itn = itn + 1;
    end
end
A = Acell{1};
for j = 2:layer
    A = A * Acell{j};
end
S = Scell{layer};
end