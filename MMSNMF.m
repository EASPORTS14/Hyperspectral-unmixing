function [A,S] = MMSNMF(X,Aini,Sini,tolerance,Tmax,option)
% ---option == 0 means using VCA-FCLS to initialize the layer
alpha = 0.001;
tao = 25;
k = 6;
layer = 10;
lowLimit = 1e-7;
P = size(Aini,2);
Acell = cell(1,layer);
Scell = cell(1,layer);
[DS,WS] = generate_graph(X,k);
[DA,WA] = generate_graph(X',k-1);
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
        ga = 0.1*aa;
        as = 2*aa;
        gs = 0.1*aa;
        Acell{i}(Acell{i}<lowLimit) = lowLimit;
        if i == 1
            Acell{i} = Acell{i} .* (X*Scell{i}'+ga*WA*Acell{i}) ./ (Acell{i}*Scell{i}*Scell{i}' + 0.5*aa*(Acell{i}.^(-0.5))+ga*DA*Acell{i});
        else
            Acell{i} = Acell{i} .* (X*Scell{i}') ./ (Acell{i}*Scell{i}*Scell{i}' + 0.5*aa*(Acell{i}.^(-0.5)));
        end
        Acell{i} = real(Acell{i});
        Scell{i}(Scell{i}<lowLimit) = lowLimit;
        Scell{i} = Scell{i} .* (Acell{i}'*X+gs*Scell{i}*WS) ./ (Acell{i}'*Acell{i}*Scell{i} + 0.5*as*(Scell{i}.^(-0.5))+gs*Scell{i}*DS);
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
%% Inner utility function
function [neighbour] = knearest(X,i,k)
   N = size(X,2);
   dis = zeros(1,N);
   neighbour = zeros(1,k);
   for j = 1:N
       if j ~= i
           dis(1,j) = norm((X(:,i)-X(:,j)),2);
       else
           dis(1,j) = 2131376484;
       end
   end
   [~,index] = sort(dis,'ascend');
   for j = 1:k
      neighbour(1,j) = index(1,j); 
   end
end
function [D,W] = generate_graph(X,k)
    N = size(X,2);
    W = zeros(N,N);
    for i = 1:N
       [neighbour] = knearest(X,i,k);
       for j = 1:k
           W(i,neighbour(1,j)) = 1;
       end
    end
    D = zeros(size(W));
    for row = 1:N
        D(row,row) = sum(W(row,:));
    end
end