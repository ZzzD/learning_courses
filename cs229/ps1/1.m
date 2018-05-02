
X = load(’logistic_x.txt’);
Y = load(’logistic_y.txt’)
X = [ones(size(X, 1), 1) X];
[theta, ll] = log_regression(X ,Y, 20);

m=size(X,1);
figure; hold on;
plot(X(Y < 0, 2), X(Y < 0, 3), ’rx’, ’linewidth’, 2);
plot(X(Y > 0, 2), X(Y > 0, 3), ’go’, ’linewidth’, 2);
x1 = min(X(:,2)):.01:max(X(:,2));
x2 = -(theta(1) / theta(3)) - (theta(2) / theta(3)) * x1;
plot(x1,x2, ’linewidth’, 2);
xlabel(’x1’);
ylabel(’x2’);
%%%%%%% log_regression.m %%%%%%%
function [theta,ll] = log_regression(X,Y, max_iters)
% rows of X are training samples
% rows of Y are corresponding -1/1 values
% newton raphson: theta = theta - inv(H)* grad;
% with H = hessian, grad = gradient
mm = size(X,1);
nn = size(X,2);
theta = zeros(nn,1);
ll = zeros(max_iters, 1);
for ii = 1:max_iters
  margins = Y .* (X * theta);
  ll(ii) = (1/mm) * sum(log(1 + exp(-margins)));
  probs = 1 ./ (1 + exp(margins));
  grad = -(1/mm) * (X’ * (probs .* Y));
  H = (1/mm) * (X’ * diag(probs .* (1 - probs)) * X);
  theta = theta - H \ grad;
end
