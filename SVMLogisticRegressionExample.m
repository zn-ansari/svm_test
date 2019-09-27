%%
%install SVM before running this example
%detailed instruction
%Installation: http://web.cvxr.com/cvx/doc/install.html
%Simple tutorial: https://web.stanford.edu/class/ee364a/lectures/cvx_tutorial.pdf

clear all
%feature vector, same as the example of Perceptron
X=[-2 1;  1 1; 1.5 -0.5; -2 -1; -1 -1.5; 2 -2];
xlabel('x_1')
ylabel('x_2')
Y=[1 1 1 -1 -1 -1]';%label
hold on
grid on %add grid to figure
axis([-3 3 -3 2])

%plot the feature vectors of class +1
scatter(X(1:3,1),X(1:3,2),200,'o','red','filled')
%plot the feature vectors of class -1
scatter(X(4:6,1),X(4:6,2),200,'s','blue','filled')




%training the SVM using CVX

cvx_begin quiet 
    variables w(2) b(1)
    minimize( norm(w))
    subject to
        1<=Y.*(X*w+b)
cvx_end
w
b
%plot the decision boundary for SVM
fplot(@(x)  -w(1)/w(2)*x-b/w(2),'red') 
%compute geometric margin for SVM
disp('Geometric Margin of SVM =' ); disp(min(Y.*(X*w+b)/norm(w)))%compute margin for SVM

%%------------Perceptron as in the example--------------
w_perceptron=[0.5,1]';%from the example of the lecture
b_perceptron=0.2;
%plot decision boundary for Perceptron
fplot(@(x)  -w_perceptron(1)/w_perceptron(2)*x-b_perceptron/w_perceptron(2),'green') 
%compute geometric margin for Perceptron
disp('Geometric Margin of Perceptron =' ); disp(min(Y.*(X*w_perceptron+b_perceptron)/norm(w_perceptron)))

%add legend to the figure.
legend('Class +1 data','Class -1 data','SVM','Perceptron')

hold off

%-------------logistic regression------------------
%change the labels to 0 and 1 (different from SVM, where we use -1 and 1 to
%denote the labels
Y=[1 1 1 0 0 0]';

%introduce the dummy feature 1 for each training sample
%w is now three dimensional with the third entry being bias b
X=[X,ones(6,1)];
%use batch gradient descent, and intialize alpha and w
alpha=0.5;
w_logistic=[0,0,0]';
while 1
    w_temp=w_logistic;
    gradient=sum((Y-1./(1+exp(-(X*w_logistic)))).*X,1)';
    w_logistic=w_logistic+alpha*gradient;
    if norm(gradient)<=1e-5
        break
    end
    
end
%output predicted label on training data set
1./(1+exp(-(X*w_logistic)))>0.5
    
%plot the decision boundary of logistic regression
fplot(@(x)  -w_logistic(1)/w_logistic(2)*x-w_logistic(3)/w_logistic(2),'blue') 

%compute geometric margin for logistic regression
Y=[1 1 1 -1 -1 -1]';
disp('Geometric Margin of Logistic Regression =' ); disp(min(Y.*(X*w_logistic)/norm(w_logistic(1:2))))%compute margin for SVM



