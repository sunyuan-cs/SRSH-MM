function [MAP,precision,recall,Precision_top,NDCG] = train_RSH( exp_data,nbits,r,gama,lambda,top_K)

X = exp_data.traindata;
yt=exp_data.testgnd;
if isvector(yt)
    Yt = sparse(1:length(yt), double(yt), 1);
    Yt = full(Yt)';
else
    Yt = yt';
end

n_anchors = 1500;

rand('seed',100);

anchor = X(randperm(size(exp_data.traingnd,1), n_anchors),:);
Dis = EuDist2(X,anchor,0);
sigma = mean(min(Dis,[],2).^0.5);
Phi_testdata = exp(-sqdist_sdh(exp_data.testdata,anchor)/(2*sigma*sigma));
Phi_traindata = exp(-sqdist_sdh(exp_data.traindata,anchor)/(2*sigma*sigma));

X=[Phi_traindata ; Phi_testdata];
data_our.indexTrain=1:size(exp_data.traindata,1);
data_our.indexTest=size(exp_data.traindata,1)+1:size(exp_data.traindata,1) + size(exp_data.testdata,1);
data_our.X=normZeroMean(X);
data_our.X=normEqualVariance(X);
data_our.label=exp_data.traingnd;

% X0=data_our.X;
% save('cifar_0.mat','X0');

% Training
fprintf('Training...\n');
% [U_logical_trn,U_logical_tst,Y]= RSH_I(data_our,nbits,r,gama, lambda);
Iter_num = 5;


N1 = length(data_our.indexTrain)'; %Ntrain
X = data_our.X(data_our.indexTrain, :)'; %train
X2 = data_our.X(data_our.indexTest, :)'; %test
y=data_our.label(data_our.indexTrain, :)';

if isvector(y)
    Y = sparse(1:length(y), double(y), 1); 
    Y = full(Y)';
else
    Y = y;
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
W1=(H1*X')/(X*X'+lambda*eye(dv));
W2=(H2*H1')/(H1*H1'+lambda*eye(size(H1,1)));
W3=(H3*H2')/(H2*H2'+lambda*eye(size(H2,1)));
tic
%-----------------------------------------------------training---------------------------------
for iter=1:Iter_num 
    fprintf('iter %d... \n',iter);
%     LapA=H3;
    % ----------------------- Hi-step -----------------------%
    H1=(W2'*W2+eye(size(W2,2)))\(W1*X+W2'*H2); 
    H2=sign(W2*H1+W3'*H3+gama*nbits*H3*Y'*Y);     
    H3_temp = (W3*H2+gama*nbits*H2*Y'*Y);
    H3=hash_const(H3_temp,nbits,N1);
    %------------------------ Wi-step -----------------------%
    W1=(H1*X')*pinv((X*X'+lambda*eye(dv)));
    W2=(H2*H1')*pinv((H1*H1'+lambda*eye(size(H1,1))));
    W3=(H3*H2')*pinv((H2*H2'+lambda*eye(size(H2,1))));


%       LapAold=H3;
%       obj(iter) = norm(LapA-LapAold,'fro')^2/norm(LapAold,'fro')^2;


   
      
      

end
    
X3=H3;
save('cifar_3.mat','X3');  

toc
      U_logical_trn=H2'>0;
      U_logical_tst=(W2*W1*X2)'>0;  
      B_compact_trn = compactbit(U_logical_trn);
      B_compact_tst = compactbit(U_logical_tst);
      DHamm = hammingDist(B_compact_tst, B_compact_trn);
      [~, orderH] = sort(DHamm, 2);
      MAP(iter)=cal_map( Y',Yt', orderH');
      [precision, recall] = precision_recall(orderH', Y',Yt');
      Precision_top_tmp = precision_at_k(orderH', Y',Yt',top_K);
      Precision_top = mean(Precision_top_tmp);
      NDCG=ndcg2_k(orderH,Y',Yt',100);
      
% save('con_nuswide.mat','obj','MAP');


end

