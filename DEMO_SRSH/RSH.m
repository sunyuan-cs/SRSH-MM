function [B_train , B_test,Y ] = RSH (data_our, nbits,r,gama,lambda)
Iter_num = 5;
N1 = length(data_our.indexTrain)'; %Ntrain
X = data_our.X(data_our.indexTrain, :)'; %train
X2 = data_our.X(data_our.indexTest, :)'; %test
y=data_our.label(data_our.indexTrain, :)';

if isvector(y)
    Y = sparse(1:length(y), double(y), 1); 
    Y = full(Y)';
else
    Y = y';
end





%---------------initialize---------------------------
%---------initialize Hi
randn('seed',100);
H1=randn(r,N1);
H2=randn(nbits,N1);
H2=H2>0;H2=H2*2-1;
H3=H2;

%---------initialize Wi
[dv,~] = size(X);
% M = zeros(dv,dv);
W1=(H1*X')/(X*X'+lambda*eye(dv));
W2=(H2*H1')/(H1*H1'+lambda*eye(size(H1,1)));
W3=(H3*H2')/(H2*H2'+lambda*eye(size(H2,1)));
%-----------------------------------------------------training---------------------------------
for iter=1:Iter_num 
    fprintf('%d...',iter);
%     LapA=H3;
    % ----------------------- Hi-step -----------------------%
    H1=(W2'*W2+eye(size(W2,2)))\(W1*X+W2'*H2); 
    H2=sign(W2*H1+W3'*H3+gama*nbits*H3*Y'*Y);     
    H3_temp = (W3*H2+gama*nbits*H2*Y'*Y);
    H3=hash_const(H3_temp,nbits,N1);

    %------------------------ Wi-step -----------------------%
    W1=(H1*X')/(X*X'+lambda*eye(dv));
    W2=(H2*H1')/(H1*H1'+lambda*eye(size(H1,1)));
    W3=(H3*H2')/(H2*H2'+lambda*eye(size(H2,1)));
    
    %------------------------ M-step -----------------------%
%      v  = sqrt(sum(W1'.*W1',2)+eps);
%      M  = diag(1./(2*v));

%       LapAold=H3;
%       obj(iter) = norm(LapA-LapAold,'fro')^2/norm(LapAold,'fro')^2;
end

% B_train=H3'>0;
% %-----------------------------------------------------Out-of-Sample-------------------------------------
% B_test=(W3*W2*W1*X2)'>0;
B_train=H2'>0;
B_test=(W2*W1*X2)'>0;
end



