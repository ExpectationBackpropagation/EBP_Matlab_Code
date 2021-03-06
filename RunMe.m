%% v8

% close all
% hold off
clear all
clc

addpath('Learning_Algorithms/');
addpath('util/');
addpath('Results/');

block_num=0; %initial block to start from - 0 generate new data set
Runs=3; %number of blocks to generate

% Learning algorithms
prompt = 'Please Choose an algorithem:\n1. Binary-EBP \n2. Real-EBP\nChosen algorithem: ';
result = input(prompt);
if result == 1
    algorithm_array={'Binary-EBP'};  % algorithms to test
    algorithms_array_alt={'Binary-EBP-P'};
elseif result == 2
    algorithm_array={'Real-EBP'};  % algorithms to test
    algorithms_array_alt={'Real-EBP-P'};
else 
    error('unknown algorithem');
end

parameter_array=1;% %eta for BackProp
w0_std=1; % standard deviation of initial conditions

% sim parameters
prompt = ['Please Choose DataSet:\n1. 20News_comp \n2. 20News_elec\n' ...
          '3. apparel_books\n4. apparel_dvd\n5. domain0 \n6. domain1\n' ...
          '7. reuters_I81_I83\n8. reuters_I654_I65\n Chosen Dataset: '];
result = input(prompt);
switch result
    case 1
        dataset='20News_comp';
    case 2
        dataset='20News_elec';
    case 3
        dataset='apparel_books';
    case 4
        dataset='apparel_dvd';
    case 5
        dataset='domain0';
    case 6
        dataset='domain1';
    case 7
        dataset='reuters_I81_I83';
    case 8
        dataset='reuters_I654_I65';
    otherwise
        error('unknown data set');
end

[M,N,size_data,~] = Get_dataset_properties(dataset);

Test_ratio=1/8; %for eight-fold cross validation
T_train=floor(size_data*(1-Test_ratio)); %total number of samples in training data
T_test=floor(size_data*Test_ratio); %total number of samples in test data

epochs=1; %number of presentations of the training set in each run
T=epochs*T_train; %total number of patterns
Rep_num=1/Test_ratio;% for K-fold cross valdiation, choose  Rep_num=1/Test_ratio, and Test_ratio=1/K;
downsample_ratio=T_train;

if (mod(T,downsample_ratio)~=0)||(mod(T_train,downsample_ratio)~=0)
    error('`downsample_ratio` must divide both `T` and `T_train`!!!!');
end

% network parameters
R=[120];  %hidden neurons
layers=[M R N];  %number of neurons in each layer
L=length(layers)-1; %number of synaptic layers


layers_string=[];

for ii=1:length(layers)
    layers_string= [layers_string [num2str(layers(ii)) 'x'] ] ; %#ok
end
layers_string(end)=[];

for rr=1:Runs %change file name in 2 places, block_num, and net+sim paramters

    block_num=block_num+1;
    
    %% Intialize

        % determined saved file name
    if length(parameter_array)==1;
        p_string=[];
    else
        p_string=['_ParameterScan_' cell2mat(algorithm_array)];
    %     name=['Task_ParameterScan_scale_' cell2mat(algorithm_array)  '_M' num2str(M) 'L' num2str(L) ];
    end
    
    if ~isempty(w0_std)
        p_string=[ p_string '_w0_std_' num2str(w0_std)];
    end
    ResultsFolder = ['Results/' dataset];
    name=['Results/' dataset '/Task_' cell2mat(algorithm_array) p_string '_layers' layers_string '_B' num2str(block_num)]; 

    if block_num>1
    % load previous run 
    L_num=length(num2str(block_num));
    name_prev=[ name(1:end-L_num) num2str(block_num-1)  ];
    temp=load([name_prev '.mat'],'Task');
    h_cell_initial=temp.Task.algorithms.h_cell;
    if isfield(temp.Task.algorithms,'bias_cell')
        bias_cell_initial=temp.Task.algorithms.bias_cell;
    end
    if isfield(temp.Task.sim_params,'T_initial')
        T_initial=temp.Task.sim_params.T+temp.Task.sim_params.T_initial;
    else
        T_initial=temp.Task.sim_params.T+1;
    end

    clear temp
    elseif block_num==1
        T_initial=1;
    end
        
    Task=struct;

    Task.algorithms.main=algorithm_array;
    Task.algorithms.alt=algorithms_array_alt;  % alternative version of algorithms - if 0, then don't check
    Task.algorithms.parameter_array=parameter_array;
    Task.algorithms.w0_std=w0_std;
    if strcmp('algorithm','MFB_BackProp')
        Task.algorithms.weight_type_cell=weight_type_cell;
    end

    Task.network_params=struct('M',M,'N',N,'L',L,'R',R,'layers',layers);   

    Task.sim_params=struct('dataset',dataset,'T',T,...
        'T_train',T_train,'T_test',T_test,'Rep_num',Rep_num,...
        'downsample_ratio',downsample_ratio,'T_initial',T_initial,'block_num',block_num);

    
    %% Intialize performance measures
    T_new=T/downsample_ratio;
    avg_train_error=zeros(T_new,length(algorithm_array),length(parameter_array));             % average training error
    best_train_error=ones(T_new,length(algorithm_array),length(parameter_array));   % best training error
    avg_train_error_alt=zeros(T_new,length(algorithm_array),length(parameter_array));             % average training error of alternative version
    best_train_error_alt=ones(T_new,length(algorithm_array),length(parameter_array));   % best training error  of alternative version

    avg_gen_error=zeros(length(algorithm_array),length(parameter_array));             % average generalization error
    all_gen_error=zeros(Rep_num,length(algorithm_array),length(parameter_array));
    best_gen_error=ones(length(algorithm_array),length(parameter_array));   % best generalization error
    avg_gen_error_alt=zeros(length(algorithm_array),length(parameter_array));             % average generalization error of alternative version    
    best_gen_error_alt=ones(length(algorithm_array),length(parameter_array));   % best generalization error  of alternative version
    all_gen_error_alt=zeros(Rep_num,length(algorithm_array),length(parameter_array));
    
    h_cell_best=cell(length(algorithm_array),length(parameter_array),Rep_num); % save the best weights for each algorithm    
    bias_cell_best=cell(length(algorithm_array),length(parameter_array),Rep_num); % save the best biases for each algorithm    

    
    %% Main Loop

    tic

    for ii=1:length(algorithm_array);  %'P','PP','BP','BPI','CP','SP','SP2'
        algorithm=cell2mat(algorithm_array(ii));  


        for jj=1:length(parameter_array)
            for rep=1:Rep_num   

                if exist('h_cell_initial','var')%initialize hidden weights         
                    h_cell=h_cell_initial{ii,jj,rep};
                else
                    h_cell=[];
                end
                
                if exist('bias_cell_initial','var') %initialize hidden bias
                    bias_cell=bias_cell_initial{ii,jj,rep};
                else
                    bias_cell=[];
                end

                if mod(T,T_train)==0
                    P=T/T_train+1;
                else
                    P=ceil(T/T_train);  
                end
                train_error=cell(P,1);                
                train_error_alt=cell(P,1);

                for pp=1:(P+1)
                    data_type='Train';
                    e=1;
                    disp('Trainning');
                    if pp==P+1
                        disp('Testing');
                        data_type='Test';
                        T_part=T_test;
                        e=0;
                    elseif pp==P
                        T_part=mod(T,T_train);
                    else
                        T_part=T_train;
                    end

              %generate data
              subset=rep;
              [ x, d ] = Generate_dataset(T_part,Test_ratio,data_type,subset,dataset);


                % Train    
                switch algorithm
                    case 'RND'
                       [r, h_cell]= Rand_network(x,L,R,N);  
                   case'Real-EBP'
                        eta=1;
                        dropout=0;
                        Task.algorithms.dropout=dropout;
                         batch_size=10;
                         Task.algorithms.batch_size=batch_size;              
                        [r, r_alt , h_cell,bias_cell]=Real_EBP_minibatch( x,d,e,layers,h_cell,bias_cell,eta,w0_std,dropout,batch_size);
                    case'Binary-EBP'
                        eta=1;
                        dropout=0;
                        Task.algorithms.dropout=dropout;
                        batch_size=10;
                        Task.algorithms.batch_size=batch_size;              
                        [r, r_alt , h_cell,bias_cell]=Binary_EBP_minibatch( x,d,e,layers,h_cell,bias_cell,eta,w0_std,dropout,batch_size);
                    otherwise
                        error('unknown learning algorithm');
                end

                    if strcmp(data_type,'Test');            
                        current_gen_error=mean(Get_errors_max(r,d));
                        current_train_error=cell2mat(train_error);
                        if cell2mat(algorithms_array_alt(ii))~='0' 
                            current_gen_error_alt=mean(Get_errors_max(r_alt,d));
                            current_train_error_alt=cell2mat(train_error_alt);
                        end
                    elseif  strcmp(data_type,'Train');
                        temp1=Get_errors_max(r,d);
                        train_error(pp)={Filtered_downample(temp1,downsample_ratio)};
                        if cell2mat(algorithms_array_alt(ii))~='0' 
                            temp2=Get_errors_max(r_alt,d);
                            train_error_alt(pp)={Filtered_downample(temp2,downsample_ratio)};
                        end
                    else
                        error('data_type must be `Test` or `Train` !!!');
                    end
                end


                avg_train_error(:,ii,jj)=avg_train_error(:,ii,jj)+current_train_error;
                avg_gen_error(ii,jj)=avg_gen_error(ii,jj)+current_gen_error;
                all_gen_error(rep,ii,jj)=current_gen_error;

                if current_gen_error<=best_gen_error(ii,jj)
                    best_gen_error(ii,jj)=current_gen_error;                    
                    best_train_error(:,ii,jj)=current_train_error;
                end   % cond breakpoint:  sum(abs(W_corr)>1)>0

                if cell2mat(algorithms_array_alt(ii))~='0'                
                    avg_train_error_alt(:,ii,jj)=avg_train_error_alt(:,ii,jj)+current_train_error_alt;
                    avg_gen_error_alt(ii,jj)=avg_gen_error_alt(ii,jj)+current_gen_error_alt;
                    all_gen_error_alt(rep,ii,jj)=current_gen_error_alt;

                if current_gen_error_alt<=best_gen_error_alt(ii,jj)
                    best_gen_error_alt(ii,jj)=current_gen_error_alt;
                    best_train_error_alt(:,ii,jj)=current_train_error_alt; 
                end
                    h_cell_array(ii,jj,rep)={h_cell};
                    bias_cell_array(ii,jj,rep)={bias_cell};
                end

            end

            avg_train_error(:,ii,jj)=avg_train_error(:,ii,jj)/Rep_num;
            avg_train_error_alt(:,ii,jj)=avg_train_error_alt(:,ii,jj)/Rep_num;
            avg_gen_error(ii,jj)=avg_gen_error(ii,jj)/Rep_num %#ok
            avg_gen_error_alt(ii,jj)=avg_gen_error_alt(ii,jj)/Rep_num %#ok
            Runtime=toc %#ok

            Task.performance=struct('avg_train_error',avg_train_error,'avg_train_error_alt',avg_train_error_alt,...
                'best_train_error',best_train_error,'best_train_error_alt',best_train_error_alt,...
                'avg_gen_error',avg_gen_error,'avg_gen_error_alt',avg_gen_error_alt,...
                'best_gen_error',best_gen_error,'best_gen_error_alt',best_gen_error_alt,...
                'all_gen_error',all_gen_error,'all_gen_error_alt',all_gen_error_alt,'Runtime',Runtime); 

             Task.algorithms.h_cell=h_cell_array;
             Task.algorithms.bias_cell=bias_cell_array;
            if ~exist(ResultsFolder,'dir')
                mkdir(ResultsFolder)
            end
            save([ name '.mat'],'Task');



       end
    end
end


%% close all

% hold all
% Plot_task(Task,0);




