function error=Get_errors_max(r,d)
% d - class, in binary coding - e.g. (+1 +1 -1 +1) denotes class 3 out of 4
% r - output

T=size(r,1);
error=zeros(T,1);

if size(d,2)==1
    for kk=1:T
        error(kk)=(sign(d(kk))~=sign(r(kk)));
        if sum(isnan(r(kk,:)))>0 %find numerical errors
            error(kk)=inf;
        end    
    end
else

    for kk=1:T
            [junk ind1]=max(d(kk,:));
            [junk ind2]=max(r(kk,:));
            error(kk)=(ind1~=ind2);
            if sum(isnan(r(kk,:)))>0 %find numerical errors
                error(kk)=inf;
            end
    end
end

end