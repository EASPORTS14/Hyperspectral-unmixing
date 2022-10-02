function [M,R] = AGMLNMF(Y,Aini,Sini,tolerance,Tmax,option)
alpha = 0.001;
k = 5;    % k-nearest neighbour unsupervised method
layer = 5;
tao = 25;
lowLimit = 1e-7;
[D,W] = generate_graph(Y,k);
disp(W);
P = size(Aini,2);
Mcell = cell(1,layer);
Rcell = cell(1,layer);
%% Handle every layer without considering the global loss(X-AS):
for i = 1:layer
    disp(['pretrain stage handling layer ',num2str(i),'......']);
    if i == 1
        Mcell{i} = Aini;
        Rcell{i} = Sini;
    else
        if option == 0
            [Ae,~,~] = VCA(Rcell{i-1},'Endmembers', P,'verbose','on');
            Se = FCLS(Rcell{i-1}, Ae);
            Mcell{i} = Ae;
            Rcell{i} = Se;
        else
            Mcell{i} = rand(P,P);
            Rcell{i} = Rcell{i-1};
        end
        Y = Rcell{i-1};
    end
    itn = 1;
    Ap = Mcell{i};
    count = 0;
    while itn < Tmax
        disp(itn);
        aa = alpha*exp(-itn/tao);
        as = 2*aa;
        eta = 0.0*aa;
        Mcell{i}(Mcell{i}<lowLimit) = lowLimit;
        Mcell{i} = Mcell{i} .* (Y*Rcell{i}') ./ (Mcell{i}*Rcell{i}*Rcell{i}' + 0.5*aa*(Mcell{i}.^(-0.5)));
        Mcell{i} = real(Mcell{i});
        Rcell{i}(Rcell{i}<lowLimit) = lowLimit;
        Rcell{i} = Rcell{i} .* (Mcell{i}'*Y+eta*Rcell{i}*W) ./ (Mcell{i}'*Mcell{i}*Rcell{i} + 0.5*as*(Rcell{i}.^(-0.5)) + eta*Rcell{i}*D);
        Rcell{i} = real(Rcell{i});
        if itn ~= 1
            err = norm(Mcell{i}-Ap);
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
        Ap = Mcell{i};
        itn = itn + 1;
    end
end
M = Mcell{1};
for j = 2:layer
    M = M * Mcell{j};
end
R = Rcell{layer}; 
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
function [D,W] = generate_graph(Y,k)
    N = size(Y,2);
    S = rand(N,N);
    D = zeros(size(S));
    L = D;
    for row = 1:N
        temp = sum(S(row,:)); 
        S(row,:) = S(row,:)/temp; 
    end
    %% generate adaptive graph
    ite = 1;
    while ite < 2
        disp(ite);
%         for i = 1:N
%             dia = 0;
%             for j = 1:N
%                 dia = dia + S(i,j) + S(j,i);
%             end
%             D(i,i) = dia/2;
%         end
%         Lnew = D - (S'+S)/2;
        for i = 1:N   % update i-th row of S
            [neighbour] = knearest(Y,i,k+1);
            tempd = 0;
            for j = 1:k
                tempd = tempd + norm((Y(:,i)-Y(:,neighbour(1,j))),2)^2; 
            end
            ga = 0.5*k*norm((Y(:,i)-Y(:,neighbour(1,k+1))),2)^2 - 0.5*tempd;
            beta = (1+ tempd/ga) / k;
            for j = 1:N
                S(i,j) = 0.5*(norm((Y(:,i)-Y(:,j)),2)^2)/ga + beta;
            end
        end
%         err = norm(L(1,:)-Lnew(1,:));
%         disp(['generate graph: ',num2str(ite),' error is: ',num2str(err)]);
        ite = ite + 1;
    end
    W = S;
    D = zeros(size(S));
    for row = 1:N
        D(row,row) = sum(W(row,:));
    end
end