function [A,S,Scell] = ML_Sparsity(X,Aini,Sini,tolerance,Tmax,option,hyper)
need_augment = 0;
if need_augment == 1     % should be set 0.1 if and only if using augment technique, otherwise it will be NAN!
    a0 = 0.1;
else
    a0 = 0.001;
end
delta = 15;          % control the ASC impact
tao = 25;
layer = 10;
super = hyper;       % the weight of superpixel
P = size(Aini,2);    % number of endmember
N = size(X,2);       % number of pixel
SW = ones(1,P);
[~,D,W] = get_matrice(X,ceil(10*sqrt(N)),0);  % number of superpixel
CD = D' + D;
CW = W' + W;
lowLimit = 1e-6;
Acell = cell(1,layer);
Scell = cell(1,layer);
%% Handle every layer without considering the global loss(X-AS):
spa = 0.25;
sps = 0.5;
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
        %spa = spa * 0.9;
        %sps = sps * 0.95;
        if need_augment == 1
            Acell{i} = rand(P,P);
            Scell{i} = Scell{i-1};
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
        augX = [X; delta*ones(1,N)];
    end
    itn = 1;
    count = 0;
    Ap = Acell{i};
    while itn < Tmax
        aa = a0*exp(-itn/tao);
        as = 2*aa;
        ba = 1*aa;
        bs = super*aa; 
        Acell{i}(Acell{i}<lowLimit) = lowLimit;
        if i > 1
            Acell{i} = Acell{i} .* (X*Scell{i}') ./ ((Acell{i}*Scell{i}*Scell{i}' + ba*(SW'*SW)*Acell{i}));
            %disp(Acell{i});
            if isnan(Acell{i}(end))
                disp('there is NAN in the A, exit(1)');
                break;
            end
        else
            Acell{i} = Acell{i} .* (X*Scell{i}') ./ (Acell{i}*Scell{i}*Scell{i}' );%+ 0.5*aa*(Acell{i}.^(spa-1)));
            Acell{i} = real(Acell{i});
        end
        Scell{i}(Scell{i}<lowLimit) = lowLimit;
        if need_augment == 1
            augA = [Acell{i}; delta*ones(1,P)];
            Scell{i} = Scell{i} .* (augA'*augX + bs*Scell{i}*CW) ./ (augA'*augA*Scell{i} + 0.5*as*(Scell{i}.^(sps-1)) + bs*Scell{i}*CD);
        else
            Scell{i} = Scell{i} .* (Acell{i}'*X + bs*Scell{i}*CW) ./ (Acell{i}'*Acell{i}*Scell{i} + 0.5*as*(Scell{i}.^(sps-1)) + bs*Scell{i}*CD);
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
function [label,D,W] = get_matrice(X,segments,local)
lowLimit = 1e-7;
[label,~,~,~] = superPixel(X,segments,1);  % we first handle the cluster problem on the original image.'1' means using primary features(RGB);
n = size(label,2);
W = zeros(n,n);
D = zeros(n,n);
for i = 1:n
    for j =1:n
        if label(1,i) == label(1,j)
            W(i,j) = 1;
            W(j,i) = 1;
        else
            W(i,j) = lowLimit;
            W(j,i) = lowLimit;
        end
    end
end
if local > 0 % use local superpixel graph
    for row = 1:n
       list = [];
       for j = 1:n
           if j ~= row && W(row,j) == 1
               distance = norm((X(:,row)-X(:,j)),2);
               list = [list;distance j];
           end
           W(row,j) = lowLimit;
       end
       [topk] = sortrows(list,1);
       for t = 1:local
          W(row,topk(t,2)) = 1; 
       end
    end
    disp(W(1,:));
end
for i = 1:n
    D(i,i) = sum(W(i,:));
end
Laplacian = D - W;
end
function [smean] = super_mean(W,S,label)
n = size(W,2);
m = size(S,1);
cluster = zeros(m,max(label));
for p = 1:n
    if cluster(1,label(1,p)) ~= 0
        continue;
    else
        sum = zeros(m,1);
        count = 0;
        for neighbour = 1:n
            if W(p,neighbour) == 1
                sum(:,1) = sum(:,1) + S(:,neighbour);
                count = count + 1;
            end
        end
        cluster(:,label(1,p)) = sum/count;
    end
end
smean = zeros(size(S));
for p = 1:n
    smean(:,p) = cluster(:,label(1,p));
end
end
%% %%%%%%%%-----------------------------------------------------