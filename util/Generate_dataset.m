function [ x, d] = Generate_dataset(T_part,test_ratio,data_type,subset,dataset)
% fetch dataset (data+labels), with:
% T_part - length of dataset to generate
% test_ratio - how much of the dataset is "test". If empty, default value is used
% subset - which subset of data to use as test, out of 1/test_ratio subsets
% data_type - do we need "Test" or "Train" data
% dataset - dataset name. must corrspond with directory name in Datasets\Classification


path = Find_dataset_path(dataset);
load(path,'data','labels','T','M','N','default_Test_ratio');
if isempty(test_ratio)
    test_ratio=default_Test_ratio;
end

if subset>1/test_ratio
    error('subset number cannot exceed 1/test_ratio!!');
end

Test_indices=round(1+(subset-1)*test_ratio*T):round(subset*test_ratio*T);
Train_indices=1:T;
Train_indices(Test_indices)=[];

if strcmp(data_type,'Test');      
    data=data(Test_indices,:);%#ok
    labels=labels(Test_indices,:); %#ok
%     if length(dataset)>10 % test only on rare events
%      if strcmp(dataset(1:10),'imbalanced')
%         Test_indices=labels>0;  
%         data=data(Test_indices,:);
%         labels=labels(Test_indices,:);      
%      end
%     end
    T0=length(labels);
 elseif  strcmp(data_type,'Train');
    data=data(Train_indices,:); %#ok
    labels=labels(Train_indices,:); %#ok
    T0=length(labels);
else
    error('data_type must be `Test` or `Train` !!!');
end


%% Randomize order   
if T_part<=T0
    temp=randperm(T0);  
    indices=temp(1:T_part);
else
    temp1=repmat(randperm(T0),1,ceil(T_part/T0));
    indices=temp1(randperm(T_part));   
end

x=data(indices,:);
d=labels(indices,:);

end