% demo for MSHNet-P classification algorithm
addpath(genpath('.\utils'))
addpath(genpath('.\dataset'))
a = load('Indian_pines_corrected.mat');
Data = a.indian_pines_corrected;
[row,col,num_feature] = size(Data);
a = load('Indian_pines_gt.mat');
Label = reshape(double(a.indian_pines_gt),row*col,1);
num_class = max(Label(:));
num_runs = 1;
OA_array = zeros(num_runs, 1);
Kappa_array = zeros(num_runs, 1);
class_accuracies_all = zeros(num_runs, num_class);
tic;
clear a;

for run = 1:num_runs
    train_sample = 0.1;
    for i = 1:num_class
        index = find(Label == i);
        train_num = max(floor(train_sample * length(index)), 1);
        train_num_array(i) = train_num;
    end
    train_num_all = sum(train_num_array);
    num_PC = 3;
    num_fuse_PC=6;
    Layernum = 7;
    
    w1=15; win_inter1 = (w1-1)/2;
    w2=31; win_inter2 = (w2-1)/2;
    w3=17; win_inter3 = (w3-1)/2;
    epsilon = 0.01;
    K=20;
    
    % The feature extraction network code will be made public after the paper is accepted.
end
elapsedTime = toc;
fprintf('The total execution time is: %.2f seconds.\n', elapsedTime);

OA_mean = mean(OA_array);
OA_std = std(OA_array);
Kappa_mean = mean(Kappa_array);
Kappa_std = std(Kappa_array);

fprintf('\n%d th (avgOA) : %.2f%%, std: %.2f\n', num_runs, OA_mean * 100, OA_std * 100);
fprintf('%d th (avgKappa) : %.4f, std: %.4f\n', num_runs, Kappa_mean, Kappa_std);