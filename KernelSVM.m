%data preprocessing
clear all

load('mnist35.mat')

n_train=length(trainy);%total number of training samples
n_test=length(testy);%total number of test samples

%show 4 training samples
% subplot(2,2,1)
% image(reshape(trainx(12,:),28,28)');
% subplot(2,2,2)
% image(reshape(trainx(992,:),28,28)');
% subplot(2,2,3)
% image(reshape(trainx(1012,:),28,28)');
% subplot(2,2,4)
% image(reshape(trainx(1112,:),28,28)');

%normalize training data
trainx=double(trainx)/255;
testx=double(testx)/255;


one=ones(n_train,1);
%compute the kernel matrix


%parameter to be tuned
sigma=10000;

K=zeros(n_train,n_train);
for i =1:1:n_train
   for j=1:1:n_train
       K(i,j)=exp(-(norm(trainx(i,:)-trainx(j,:)))^2/(2*sigma^2));
   end
end

cvx_begin quiet 
    variables a(n_train)
    minimize( -one'*a+1/2*(trainy.*a)'*K*(trainy.*a) )
    subject to
        0<=a
        0==a'*trainy
cvx_end

temp=(a.*trainy)' * K ;

b=- ( max(temp(find(trainy==-1)))  + min(temp(find(trainy==1))))/2;

predicted_y=(sign((a.*trainy)'*K+b))';
% sum(predicted_y~=trainy);
disp('Training loss:')
disp(sum(predicted_y~=trainy));

%kernel X
kX=zeros(n_train,n_test);
for i=1:n_train
    for j=1:n_test
       kX(i,j)=exp(-(norm(trainx(i,:)-testx(j,:)))^2/(2*sigma^2));
    end
end

p_test=sign((a.*trainy)'*kX+b);
disp('Test loss:')
disp(sum(p_test~=testy')/n_test)




