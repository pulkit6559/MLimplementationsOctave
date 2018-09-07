function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
  c = (X*theta - y);
  J = (1.0/(2.0*m))*(c'*c + (theta(2:end, :)'*theta(2:end, :))*lambda);
  
    %grad = (X'*c + [theta(1);(theta(2:end).*lambda)]).*(1.0/((1.0)*m));
  
  reg = [0; theta(2:end, :).*((lambda*1.0)/(1.0*m))];
  grad = (1.0/m)*(X'*c) + reg;
% =========================================================================
end
