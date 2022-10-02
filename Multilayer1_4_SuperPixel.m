function [A,S] = Multilayer1_4_SuperPixel(X,Aini,Sini,tolerance,Tmax,option,hyper)
need_augment = 0;
if need_augment == 1     % should be set 0.1 if and only if using augment technique, otherwise it will be NAN!
    a0 = 0.1;
    delta = 15;          % control the ASC impact
else
    a0 = 0.001;
end
tao = 25;
layer = hyper;
super = 0.08;
P = size(Aini,2);    % number of endmember
N = size(X,2);       % number of pixel
[~,D,W] = get_matrice(X,ceil(1*sqrt(N)));
CD = D' + D;
CW = W' + W;
lowLimit = 1e-6;
Acell = cell(1,layer);
Scell = cell(1,layer);
%% Handle every layer without considering the global loss(X-AS):
for i = 1:layer
    disp(['pretrain stage handling layer ',num2str(i),'......']);
    if i == 1
        Acell{i} = Aini;
        if need_augment == 1
            Xb = [X; delta*ones(1,N)];
            Ab = [Aini; delta*ones(1,P)];
            Scell{i} = (Ab'*Ab)^-1*Ab'*Xb;
            augX = [X; delta*ones(1,N)];
        else
            Scell{i} = Sini;
        end
    else
        if need_augment == 1
            Acell{i} = rand(P,P);
            Scell{i} = Scell{i-1};
            augX = [X; delta*ones(1,N)];
        else
            if option == 0
                [Ae,~,~] = VCA(Scell{i-1},'Endmembers', P,'verbose','on');
                Se = FCLS(Scell{i-1}, Ae);
            else
                Ae = rand(P,P);
                Se = Scell{i-1};
            end
            Acell{i} = Ae;
            Scell{i} = Se;
        end
        X = Scell{i-1};
    end
    itn = 1;
    count = 0;
    Ap = Acell{i};
    while itn < Tmax
        aa = a0*exp(-itn/tao);
        as = 2*aa;
        bs = super*aa; 
        Acell{i}(Acell{i}<lowLimit) = lowLimit;
        Acell{i} = Acell{i} .* (X*Scell{i}') ./ (Acell{i}*Scell{i}*Scell{i}' + 0.5*aa*(Acell{i}.^(-0.75)));
        Acell{i} = real(Acell{i});
        Scell{i}(Scell{i}<lowLimit) = lowLimit;
        if need_augment == 1
            augA = [Acell{i}; delta*ones(1,P)];
            Scell{i} = Scell{i} .* (augA'*augX + bs*Scell{i}*CW) ./ (augA'*augA*Scell{i} + 0.5*as*(Scell{i}.^(-0.5)) + bs*Scell{i}*CD);
        else
            Scell{i} = Scell{i} .* (Acell{i}'*X + bs*Scell{i}*CW) ./ (Acell{i}'*Acell{i}*Scell{i} + 0.5*as*(Scell{i}.^(-0.5)) + bs*Scell{i}*CD);
        end
        Scell{i} = real(Scell{i});
        if itn ~= 1
            err = norm(Acell{i}-Ap);
            disp([num2str(err) ,' @ ', num2str(itn)])
            if(err < tolerance)
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
%% inner utility function
function [Laplacian,D,W] = get_matrice(X,segments)
[label,~,~,~] = superPixel(X,segments,1);  % we first handle the cluster problem on the original image.'1' means using primary features
n = size(label,2);
W = zeros(n,n);
D = zeros(n,n);
for i = 1:n
    for j =1:n
        if label(1,i) == label(1,j)
            W(i,j) = 1;
            W(j,i) = 1;
        else
            W(i,j) = 1e-7;
            W(j,i) = 1e-7;
        end
    end
end
for i = 1:n
    D(i,i) = sum(W(i,:));
end
Laplacian = D - W;
end
