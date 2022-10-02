clc; clear all; close all;
%% Algorithm Parameters:
Tmax = 400;
data_source = 3;
tolerance = 1e-6;
no_abundance = 0;
need_initialize = 0;
plot_single_endmember = 1;
%% Load the whole USGS spectral library:
load('Spectral_library.mat');
slib = spectral_library;
col_index = [];
for col = 1:size(slib,2)
    count = 0;
    for row = 1:size(slib(:,col),1)
       if slib(row,col) == 0
           count = count + 1;
       end
    end
    if count < 3
        col_index = [col_index col]; 
    end
end
Alib = zeros(size(slib,1),size(col_index,2));
for col = 1:size(col_index,2)
    Alib(:,col) = slib(:,col_index(col));
end
%% Load Synthetic or Real World Data
if floor(data_source) == 1
    load syntheticData;
%    %% Add Noise
%     SNR = 35;    % signal noise ratio
%     variance = sum(mixed(:).^2)/10^(SNR/10)/M/N/D;  % (hyperparametres are stored in the .mat file)
%     n = sqrt(variance) * randn([D M*N]);
%     mixed = mixed' + n;
%     clear n;
%     X_40 = mixed;
%     save('syntheticData','X_40','-append');
    need_initialize = 1;
    switch data_source
        case 1.1
            X = X_40;
            Ae = Ae_VCA_FCLS40;
            Se = Se_VCA_FCLS40;
        case 1.2
            X = X_35;
            Ae = Ae_VCA_FCLS35;
            Se = Se_VCA_FCLS35;
        case 1.3
            X = X_30;
            Ae = Ae_VCA_FCLS30;
            Se = Se_VCA_FCLS30;
        case 1.4
            X = X_25;
            Ae = Ae_VCA_FCLS25;
            Se = Se_VCA_FCLS25;
        case 1.5
            X = X_20;
            Ae = Ae_VCA_FCLS20;
            Se = Se_VCA_FCLS20;
        case 1.6
            X = X_15;
            Ae = Ae_VCA_FCLS15;
            Se = Se_VCA_FCLS15;
         case 1.7
            X = X_10;
            Ae = Ae_VCA_FCLS10;
            Se = Se_VCA_FCLS10;
         case 1.8
            X = X_5;
            Ae = Ae_VCA_FCLS5;
            Se = Se_VCA_FCLS5;
    end
   %% remove noise
%     [UU,SS,VV] = svds(X,P);  % singular value decomposition(choose top P)
%     X = UU * UU' * X;
    %X = X * 1e4;  % we may rescale the original dataset to mock the dataset like Cuprite or Jasper!
else
    if data_source == 2
        load Samson_end;
        load Samson_hsi;
        P = 3;
        X = V;
        Ae = Ae_VCA_FCLS;
        Se = Se_VCA_FCLS;
        Ao = M;
        abf = A;
        Alib = Alib(3:158,:);  % Samson only has 156 bands!
        wlen = temp(3:158,1);
        name = cood;
        %X = X * 1e4;  % we may rescale the original dataset to mock the Jasper!
    else
        if data_source == 3
            load Jasper_end;
            load JasperRidge2_F224_2.mat
            P = 4;
            Ao = M;
            abf = A;
            Ae = Ae_ML_Sparsity;
            Se = Se_ML_Sparsity;
            name = cood;
            X = zeros(198,size(Y,2));
            tlib = zeros(198,size(Alib,2));
            wlen = zeros(198,1);
            for s = 1:198
                wlen(s,1) = temp(SlectBands(s,1),1);
                X(s,:) = Y(SlectBands(s,1),:);
                tlib(s,:) = Alib(SlectBands(s,1),:);
            end
            Alib = tlib;
            %X = X / 3.5e4;   % the original image may have something wrong?
        else
            load Cuprite12_groundTruth.mat;
            load Cuprite_R188.mat;
            P = 12;
            no_abundance = 1;
            abf = zeros(P,size(X,2));
            need_initialize = 1;
            Ae = Ae_ML_Sparsity;
            Se = Se_ML_Sparsity;
            name = cood;
            tlib = zeros(188,size(Alib,2));
            wlen = zeros(188,1);
            Ao = zeros(188,P);
            for s = 1:188
                Ao(s,:) = M(s,:);
                wlen(s,1) = waveLength(1,SlectBands(s,1));
                tlib(s,:) = Alib(SlectBands(s,1),:);
            end
            Alib = tlib;
            %X = X/100;
        end
    end   
end
%% Using VCA to initialize the signature. Note that FCLS for abundance may in each algorithm.
if need_initialize == 1
    [Ae,index,~] = VCA(X,'Endmembers', P,'verbose','on');
    Se = FCLS(X, Ae);
end
disp('VCA end! All endmembers have been found! We need to use similarity matrix to match them one by one.');
disp('FCLS end! Abundance matrix has been solved! We also need to repermutate them to adapt to the standard one.');
%% NMF algorithms comparison: parameters matter a lot! Need modification.
tic                                              % 'e' means estimator Ae means signature Se means abundance
%[Ae,Se] = L1_2NMF(X,Ae,tolerance,Tmax,0,Se); 
%[Ae,Se] = MVCNMF(X,P,tolerance,Tmax,1,Ae,Se);
%[Ae,Se] = MDMD_NMF(X,Ae,Se,Tmax,tolerance);
%[Ae,Se] = multilayerNMF(X,P,Ae,Tmax,tolerance); 
%[Ae,Se] = L1_4MLNMF(X,Ae,Se,tolerance,Tmax,0);
%[Ae,Se] = S2NMF(X,Ae,Se,tolerance,Tmax);
%[Ae,Se] = L1_DNMF(X,Ae,Se,Tmax);
%[Ae,Se] = SDNMF(X,Ae,Se,Tmax);
%[Ae,Se] = SDNMF_TV(X,Ae,Se,Tmax);
%[Ae,Se] = GLNMF(X,Ae,Se,Tmax);
%[Ae,Se] = SpNMFB(X,Ae,Se,tolerance,Tmax);        
%[Ae,Se] = SpNMFP(X,Ae,Se,tolerance,Tmax);
%[Ae,Se] = Multilayer1_4_SuperPixel(X,Ae,Se,tolerance,Tmax,0,10);
%[Ae,Se] = ML_Super_NMFrefine(X,Ae,Se,tolerance,Tmax,0,0.08);
%[Ae,Se] = AGMLNMF(X,Ae,Se,tolerance,Tmax,0);
%[Ae,Se] = MMSNMF(X,Ae,Se,tolerance,Tmax,0);
%[Ae,Se,Scell] = ML_Sparsity(X,Ae,Se,tolerance,Tmax,0,0.08);
%% Sparse regression: need spectral library(But how many kinds of essence should be chosen?!)
%[Ae,Se] = L2_1NMF(X,Alib,P,tolerance,Tmax);
%[Ae,Se] = SSLRSU(X,Alib,P,tolerance,Tmax);           
%[Ae,Se] = LSSP_RSU(X,Alib,P,tolerance,Tmax);  
%[opt] = autotest(@ML_Sparsity,P,X,Ae,Se,Ao,abf,Tmax,tolerance,no_abundance,0.037,0.2);
toc
%% Save the result of Ae and Se in the corresponding dataset_end.mat files
% algorithm_name = 'ML_Sparsity';
% dataset1 = 'syntheticData.mat';
% dataset2 = 'Samson_end.mat';
% dataset3 = 'Jasper_end.mat';
% dataset4 = 'Cuprite_R188.mat';
% eval(['Ae_',algorithm_name,'=','Ae',';']);
% eval(['Se_',algorithm_name,'=','Se',';']);
% eval(['destination=','dataset',num2str(floor(data_source)),';']);
% eval(['save(''',destination,''',''Ae_',algorithm_name, ''',''-append'');' ]);
% eval(['save(''',destination,''',''Se_',algorithm_name, ''',''-append'');' ]);
%% Permutation and Rescaling(that is to say:normalization for Ae and Se)
perm = permute_corr(Ao,Ae);
disp(perm);
Ae = Ae * perm;              % repermulate the row
Ae = Ae ./ repmat(max(Ae), size(Ae,1), 1);
Ae = Ae .* repmat(max(Ao), size(Ae,1), 1);
if no_abundance == 0
    Se = perm' * Se;             % repermulate the row
    col_nor = sum(Se,1);
    col_nor(col_nor == 0) = 1;
    Se = Se ./ (ones(P,1)*col_nor);
end
% load('layer_sparsity.mat');
% for layer = 1:10
%     Se = Scell{layer};
%     col_nor = sum(Se,1);
%     col_nor(col_nor == 0) = 1;
%     Se = Se ./ (ones(P,1)*col_nor);
%     count0 = 0;
%     for row = 1:size(Se,1)
%         for col = 1:size(Se,2)
%             if Se(row,col) < 1e-2
%                 count0 = count0 + 1;
%             end
%         end
%     end
%     disp(count0);
% end
%% Evaluate: using SAD & AAD
[sAD,rms] = ADnew(Ao,Ae,'A');
disp(sAD)
disp(mean(sAD));
disp(['rmsSAD is: ',num2str(rms)]);
if no_abundance == 0
    [aAD,rms] = ADnew(abf,Se,'S');
    disp(['rmsAAD is: ',num2str(rms)]);
end
%% Plot signatures matrix
if plot_single_endmember == 0
    figure(1); 
    %title('端元光谱性质曲线');
    hold on;
    plot(wlen,Ao(:,1:P),'LineWidth',2.25);
    plot(wlen,Ae(:,1:P),'-.','LineWidth',1.25);
    xlabel('波长(\mum)');
    ylabel('反射率');
    legend(name);
    axis([wlen(1) wlen(end) 0 1]);
else
   for f = 1:P
      figure(f+3);
      plot(wlen,Ao(:,f),'LineWidth',2.25);
      hold on;
      plot(wlen,Ae(:,f),'-.','LineWidth',1.25);
      xlabel('wavelength(\mum)');
      ylabel('reflectance');
      SB = strcat(name(f,1),'-truth');
      legend([SB,'-estimator']);
      axis([wlen(1) wlen(end) 0 1]);
   end
end
%% Plot abundance matrix for one endmember using heatmap
if no_abundance == 0
    N = size(abf,2);
    edge = sqrt(N);
    otree = zeros(edge,edge);
    etree = zeros(edge,edge);
    Se(:,1) = ones(P,1);     % To normalize the measurement of heatmap.
    substance_type = 3;
    for row = 1:edge
        otree(row,:) = abf(substance_type,1+edge*(row-1):(edge*row));
        etree(row,:) = Se(substance_type,1+edge*(row-1):(edge*row));
    end
    figure(2);
    heatmap(otree);
    colormap(gca, 'jet');
    %title('');
    figure(3);
    heatmap(etree);
    colormap(gca, 'jet');
    title('Abundance heatmap of estimator');
end
%% ------------------------------------------------------------------------