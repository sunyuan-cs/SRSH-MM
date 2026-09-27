close all;
clear all;


load place205
%  load('C:/Users/sunyu/Desktop/AAAI_demo/dataset/place205_noise_0.1.mat')
exp_data.traingnd=train_label_place205;
exp_data.testgnd=test_label_place205;
cateTrainTest = bsxfun(@eq, train_label_place205, test_label_place205');%similarity
exp_data.traindata = double(train_data_place205);
exp_data.testdata = double(test_data_place205);
exp_data.WTT=cateTrainTest';

preci=[];
total_res=[];
top_K=50;
nbits_set =[8 16 32 48 64];% 6 8 10 
candidate_gama= [0.01];
candidate_lambda=[1e-5];

% candidate_gama= [1e-3 1e-2 1e-1 1 1e1 1e2 1e3];
% candidate_lambda=[1e-5 1e-4 1e-3 1e-2 1e-1 1];
for ii=1:length(nbits_set)
    for c_gama = 1: length(candidate_gama)
        for c_lambda = 1: length(candidate_lambda)
            paras.gama = candidate_gama(c_gama);
            paras.lambda = candidate_lambda(c_lambda);
            nbits=nbits_set(ii);
            candidate_r= [nbits];
            for c_r = 1: length(candidate_r)
                paras.r = candidate_r(c_r)
                [MAP,precision,recall,Precision_top,NDCG] =train_RSH(exp_data, nbits,paras.r , paras.gama,paras.lambda,top_K);
                fprintf('MAP result of RHL: %d...   \n', MAP );
                total_res = [total_res; MAP,Precision_top,NDCG,paras.r,paras.gama, paras.lambda, nbits];
                index=[1,101,201,301,401,501,601,701,801,901,1001];
                precision=precision(index,:);
                preci=[preci,precision];
            end
            
        end
    end
    
end
% save('pre_rec_place.mat','preci');

save('result__place.mat','total_res');

