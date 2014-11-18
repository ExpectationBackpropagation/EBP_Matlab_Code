function  downsampled_x = Filtered_downample( x,downsample_ratio )
%FILTERED_DOWNAMPLE Summary of this function goes here
%   Detailed explanation goes here


    %filter
%     [B,A] = maxflat(10,10,1/downsample_ratio);
%     temp = filtfilt(B,A,x);

    if length(x)==downsample_ratio
        downsampled_x=mean(x);        
    else
        temp=smooth(x,downsample_ratio);
        % downsample 
        downsampled_x = downsample(temp,downsample_ratio );  
    end

end

