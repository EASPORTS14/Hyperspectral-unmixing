function [X,W] = SpNMFB(Y,Aini,Sini,tolerance,Tmax)
X = Aini;
W = Sini;
band = size(X,1);
delta = 0.05;
k1 = 0.5;
k2 = 0.2;
itenum = 0;
lb = zeros(band,1);
U = zeros(band,band);
%% begin iteration
while itenum<Tmax
      itenum = itenum + 1;
      disp(itenum);
      err = Y - X*W;
      for i=1:band
          lb(i,1) = norm(err(i,:),2)^2; 
      end
      ls = sort(lb,'ascend');
      kk = k1+(itenum-1)*delta;
      if kk>1
          kk=1;
      end
      g1 = ls(floor(kk*band),1);
      g2 = ls(floor(k2*band),1);
      for i=1:band
          [u] = self_pace2(lb(i,1),g1,g2);
          U(i,i) = u; 
      end
      Ye = U.^(0.5) * Y;
      Xe = U.^(0.5) * X;
      [Xe,W] = L1_2NMF(Ye,Xe,tolerance,20,1,W);
      for i=1:band
        if U(i,i) ~=0
            U(i,i) = U(i,i)^(-0.5);
        end
      end
      X = U * Xe;
end
end
%% inner utility function
function [u] = self_pace1(lt,g1,g2)
cau = (g1*g2)/(g1-g2);
if lt<g2
   u = 1; 
else 
    if lt>g1
        u = 0.0001;
    else
        u = cau*(g1-lt)/(g1*lt);
    end
end
end
function [u] = self_pace2(lt,g1,g2)
cau = (g1*g2)/(g1-g2);
if lt<g2
   u = lt; 
else 
    if lt>g1
        u = g2 + cau*(log(lt/g2)+g2/g1-1);
    else
        u = g2 + cau*(log(lt/g2)+g2/g1-lt/g1);
    end
end
end   