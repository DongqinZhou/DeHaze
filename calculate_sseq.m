% SSEQ: https://www.sciencedirect.com/science/article/pii/S0923596514000927
% SSEQ release: http://live.ece.utexas.edu/research/quality/SSEQ_release.zip
% install libsvm for matlab in ubuntu is straightforward.
% helpful links if you encounter the same problem as I did: https://alexxunxu.wordpress.com/2018/01/15/version-cxxabi_1-3-8-not-found/ https://www.mathworks.com/matlabcentral/answers/329796-issue-with-libstdc-so-6

clc;
clear;
path = ''; % path where images are stored
files = dir(path);
nb_files = size(files, 1);
nb_images = nb_files - 2;
sseq(nb_images) = 0;
for i = 3:nb_files
    fprintf('Processing image %d\n', i-2);
    image = imread([path, '/', files(i).name]);
    fprintf('Image size is %d\n', size(image));
    sseq(i-2) = SSEQ(image);
end

fprintf('Mean of SSEQ is %5.2f\n', mean(sseq));