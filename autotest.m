function [opt_c] = autotest(algorithm,P,X,Aini,Sini,Ao,abf,Tmax,tolerance,no_abundance,sad,aad)
opt_c = [0,0,0];
save('opt_para.mat','opt_c');
hyper = 1;
while 1
 %% using uniform initialization matrix A AND S
    [Ae,~,~] = VCA(X,'Endmembers', P,'verbose','on');
    Se = FCLS(X, Ae);
    Aini = Ae;
    Sini = Se;
    [Ae,Se] = algorithm(X,Aini,Sini,tolerance,Tmax,0,hyper);
    %% Permutation and Rescaling(that is to say:normalization for Ae and Se)
    perm = permute_corr(Ao,Ae);
    Ae = Ae * perm;              % repermulate the row
    Ae = Ae ./ repmat(max(Ae), size(Ae,1), 1);
    Ae = Ae .* repmat(max(Ao), size(Ae,1), 1);
    if no_abundance == 0
        Se = perm' * Se;             % repermulate the row
        col_nor = sum(Se,1);
        col_nor(find(col_nor==0)) = 1;
        Se = Se ./ (ones(P,1)*col_nor);
    end
    %% Evaluate: using SAD & AAD
    [sAD] = ADnew(Ao,Ae,'A');
    if no_abundance == 0
        [aAD] = ADnew(abf,Se,'S');
    else
        aAD = [0 0 0];
    end
    load('opt_para.mat');
    temp = opt_c;
    opt_c = [temp;hyper mean(sAD) mean(aAD)];
    save('opt_para.mat','opt_c');
    if mean(sAD)<sad && mean(aAD)<aad
        disp('already find the optimal hyper_parametre!');
        break;
    else
        hyper = hyper * 0.95;
    end
end
end