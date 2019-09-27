%data preprocessing
clear all

load('mnist35.mat')

n_train=length(trainy);%total number of training samples
n_test=length(testy);%total number of test samples

%show 4 training samples
subplot(2,2,1)
image(reshape(trainx(12,:),28,28)');
subplot(2,2,2)
image(reshape(trainx(992,:),28,28)');
subplot(2,2,3)
image(reshape(trainx(1012,:),28,28)');
subplot(2,2,4)
image(reshape(trainx(1112,:),28,28)');

%normalize training data
trainx=double(trainx)/255;
testx=double(testx)/255;


one=ones(n_train,1);

X=trainx*trainx';

cvx_begin quiet 
    variables a(n_train)
    minimize( -one'*a+1/2*(trainy.*a)'*X*(trainy.*a) )
    subject to
        0<=a
        0==a'*trainy
cvx_end

temp=(a.*trainy)' * X ;

b=- ( max(temp(find(trainy==-1)))  + min(temp(find(trainy==1))))/2;

predicted_y=(sign((a.*trainy)'*X+b))';

disp('Training loss:')
disp(sum(predicted_y~=trainy));

crossX=trainx*testx';

p_test=sign((a.*trainy)'*crossX+b);
disp('Test loss:')
disp(sum(p_test~=testy')/n_test)




