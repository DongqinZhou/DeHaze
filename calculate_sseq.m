clc;
clear;
path = '/home/jianan/Incoming/dongqin/Test_RTTS_Results/Real_MSCNN';
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