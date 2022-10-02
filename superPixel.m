function [label_matrix, numlabels, seedx, seedy] = superPixel(X,numSuperpixels,pca)
[band,pixel] = size(X);
nl = sqrt(pixel);
ns = nl;
if pca == 1
    n_pc = 4;
    PC = fPCA_2D_SpecificV1(X',1,1,0);
    PC = PC(:,1:n_pc);
    HIM = reshape(PC,nl,ns,n_pc);
else
    n_pc = band;
end
% reshape the data into a vector
input_img = zeros(1, nl * ns * n_pc);
startpos = 1;
for i = 1 : nl % lines
    for j = 1 : ns % columes
        if pca == 0
            input_img(1,startpos:startpos + n_pc - 1) = X(:,(i-1)*ns+j)'; % bands
        else
            input_img(startpos : startpos + n_pc - 1) = HIM(i, j, :); % bands
        end
        startpos = startpos + n_pc;
    end
end

%% SLIC Segmentation
%numSuperpixels = 25;  % the desired number of superpixels
compactness = 0.1; % compactness2 = 1-compactness, the clustering is according to: compactness*dxy+compactness2*dspectral
dist_type = 2; % distance type - 1:Euclidean£»2£ºSAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: all the pixles participate in the clustering£¬ 2: some pixels would not
% labels: segmentation results
% numlabels: the final number of superpixels
% seedx: the x indexing of seeds
% seedy: the y indexing of seeds
[labels, numlabels, seedx, seedy] = hybridseg(input_img, nl, ns, n_pc, numSuperpixels, compactness, dist_type, seg_all);
% numlabels is the same as number of superpixels
label_matrix = labels;
disp('Superpixel segment end! Then we can computer some fixed matrix.');
end