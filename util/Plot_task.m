function Plot_task(Task,show_best)
    
     set(0,'defaultlinelinewidth',2)

    window_avg=1;
    window_best=1;
    T=Task.sim_params.T;
    A=length(Task.algorithms.main); %number of main algorithms;
    
    avg_train_error=Task.performance.avg_train_error;
    avg_train_error_alt=Task.performance.avg_train_error_alt;
    best_train_error=Task.performance.best_train_error;
    best_train_error_alt=Task.performance.best_train_error_alt;
    
    avg_gen_error=Task.performance.avg_gen_error%#ok
    avg_gen_error_alt=Task.performance.avg_gen_error_alt%#ok
   if show_best
    best_gen_error=Task.performance.best_gen_error%#ok
    best_gen_error_alt=Task.performance.best_gen_error_alt %#ok
   end
    dt=Task.sim_params.downsample_ratio;
    if isfield(Task.sim_params,'T_initial')
        t0=Task.sim_params.T_initial;
    else
        t0=0;
    end
    
    t=t0+(1:dt:T);
    lgnd={};
    
    if (~isfield(Task.algorithms,'parameter_array'))||(length(Task.algorithms.parameter_array)==1);
     set(0,'DefaultAxesColorOrder',jet(2*A));
    for ii=1:A
    figure(1)
    plot(t,smooth(avg_train_error(:,ii),window_avg));
    hold all
    lgnd(end+1)=Task.algorithms.main(ii);%#ok
    
    if cell2mat(Task.algorithms.alt(ii))~='0'
        plot(t,smooth(avg_train_error_alt(:,ii),window_avg),'--');
        hold all
        lgnd(end+1)=Task.algorithms.alt(ii); %#ok       
    end
    
    if show_best
    figure(2)
    plot(t,smooth(best_train_error(:,ii),window_best));
    hold all
    
    if cell2mat(Task.algorithms.alt(ii))~='0'
        plot(t,smooth(best_train_error_alt(:,ii),window_best),'--');
        hold all
    end
    end 
    
    end
    
    % perfect_generalization_threshold1=1.45*M; %Lyuu and Rivin
    % perfect_generalization_threshold2=pi*M*log(M); %Fang and Venkatesh
    % (?)

    figure(1)
    % line(perfect_generalization_threshold1*[1 1],[0 1],'Color','k','Linestyle',':');
    % line(perfect_generalization_threshold2*[1 1],[0 1],'Color','k');

    legend(lgnd);
    % legend([ repmat('\sigma_0=',length(parameter_array),1) num2str(parameter_array')]);
    xlabel('t');
    ylabel('error');
    title('average training error');

    if show_best  
        figure(2)
        % line(perfect_generalization_threshold1*[1 1],[0 1],'Color','k','Linestyle',':');
        % line(perfect_generalization_threshold2*[1 1],[0 1],'Color','k');
        legend(lgnd);
        ylabel('error');
        % legend([ repmat('\sigma_0=',length(parameter_array),1) num2str(parameter_array')]);
        xlabel('t');
        title('best training error');
    end
    
    else
        parameter_array=Task.algorithms.parameter_array;
         set(0,'DefaultAxesColorOrder',hsv(length(parameter_array)))
         
    for ii=1:length(parameter_array)
        figure(1)
        plot(t,smooth(avg_train_error(:,1,ii),window_avg));
        hold all
        
        figure(2)
        if cell2mat(Task.algorithms.alt)~='0'
            plot(t,smooth(avg_train_error_alt(:,1,ii),window_avg));
            hold all      
        end

        if show_best
        figure(3)
        plot(t,smooth(best_train_error(:,1,ii),window_best));
        hold all

        figure(4)   
        if cell2mat(Task.algorithms.alt)~='0'
            plot(t,smooth(best_train_error_alt(:,1,ii),window_best));
            hold all
        end
        end
    end

        % perfect_generalization_threshold1=1.45*M; %Lyuu and Rivin
        % perfect_generalization_threshold2=pi*M*log(M); %Fang and Venkatesh
        % (?)
        
        K=2;
        if show_best
            K=4;
        end
        
        for kk=1:K
            figure(kk)
            % line(perfect_generalization_threshold1*[1 1],[0 1],'Color','k','Linestyle',':');
            % line(perfect_generalization_threshold2*[1 1],[0 1],'Color','k');
            legend([ repmat('$\eta$ =',length(parameter_array),1) num2str(parameter_array')]);
            xlabel('t');
            ylabel('error');
        end
        
        figure(1)
        title(['average training error - ' cell2mat(Task.algorithms.main)]);
        figure(2)        
        title(['average training error - ' cell2mat(Task.algorithms.alt)]);
        if show_best
        figure(3)
        title(['best training error - ' cell2mat(Task.algorithms.main)]);
        figure(4)
        title(['best  training error - ' cell2mat(Task.algorithms.alt)]);
        end
    end 

end 