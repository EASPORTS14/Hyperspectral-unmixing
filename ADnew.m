function [ad,rms] = ADnew(W,We,type)
if type=='A'
    m = size(W,2);
    ad = zeros(1,m);
    temp = 0;
    for i=1:m 
        ad(1,i) = acos(W(:,i)'*We(:,i)/(norm(W(:,i)')*norm(We(:,i))));
        temp = temp + ad(1,i)*ad(1,i);
    end
    rms = sqrt(temp/m);
else
    if type=='S'
        N = size(W,2);
        ad = zeros(1,N);
        temp = 0;
        for i=1:N
            %ad(1,i) = acos( W(i,:)*We(i,:)'/(norm(W(i,:))*norm(We(i,:)') ) );
            ad(1,i) = acos( W(:,i)'*We(:,i)/(norm(W(:,i)')*norm(We(:,i)) ) );
            temp = temp + ad(1,i)*ad(1,i);
        end
        rms = sqrt(temp/N);
    end
end
end