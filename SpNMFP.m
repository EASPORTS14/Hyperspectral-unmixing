function [X,W] = SpNMFP(Y,Aini,Sini,tolerance,Tmax)
X = Aini;
W = Sini;
pixel = size(Y,2);
delta = 0.05;
k1 = 0.5;
k2 = 0.2;
itenum = 0;
lb = zeros(pixel,1);
U = zeros(pixel,pixel);
%% begin iteration
while itenum < Tmax
      itenum = itenum + 1;
      disp(itenum);
      err = Y - X*W;
      for i=1:pixel
          lb(i,1) = norm(err(:,i),2)^2; 
      end
      ls = sort(lb,'ascend');
      kk = k1+(itenum-1)*delta;
      if kk>1
          kk=1;
      end
      g1 = ls(floor(kk*pixel),1);
      g2 = ls(floor(k2*pixel),1);
      for i=1:pixel
          [u] = self_pace2(lb(i,1),g1,g2);
          U(i,i) = u; 
      end
      disp(diag(U));
      Ye = Y * (U.^(0.5));
      We = W * (U.^(0.5));
      [X,W] = L1_2NMF(Ye,X,tolerance,20,1,We);
      for i = 1:pixel
        if U(i,i) ~= 0
            U(i,i) = U(i,i)^(-0.5);
        end
      end
      W = W * U;
end
end
%% inner utility function
function [u] = self_pace1(lt,g1,g2)  % May have something wrong with the diagnal value.
cau = (g1*g2)/(g1-g2);
if lt<g2
   u = 1; 
else 
    if lt>g1
        u = 0.01;
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