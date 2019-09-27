%data preprocessing
clear all

load('mnist35.mat')

n_train=length(trainy);%total number of training samples
n_test=length(testy);%total number of test samples

m_data=size(trainx,2);
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


cvx_begin quiet 
    variables w(m_data) b(1)
    minimize( norm(w))
    subject to
        1<=trainy.*(trainx*w+b)
cvx_end




predicted_y=sign(trainx*w+b);

disp('Training loss:')
disp(sum(predicted_y~=trainy));



p_test=sign(testx*w+b);
disp('Test loss:')
disp(sum(p_test~=testy)/n_test)




