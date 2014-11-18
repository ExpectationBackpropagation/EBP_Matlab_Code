function [M,N,T,default_Test_ratio] = Get_dataset_properties(dataset)
% obtain :
% M - number of attributes
% N - number of outputs (e.g., classes)
% T - total number of samples

path = Find_dataset_path(dataset);

load(path,'T','M','N','default_Test_ratio');


end