function data_file_path = Find_dataset_path(dataset)

if ~ischar(dataset)
    error('not a string!');
end

Directory='Datasets/Classification/';
data_file_path=[ Directory dataset '/' 'processed_data.mat'];

if length(dataset)>10
    if strcmp(dataset(1:10),'imbalanced')
       num_string= dataset(11:end);
       data_file_path=[ Directory 'imbalanced' '/' 'processed_data' num_string '.mat']; 
    end
end

if length(dataset)>5
    if strcmp(dataset(1:5),'mnist')
       string= dataset(6:end);
       data_file_path=[ Directory 'mnist' '/' 'processed_data' string '.mat']; 
    end
end
                
if ~exist(data_file_path,'file')
    error('unknown dataset!!');
end

end