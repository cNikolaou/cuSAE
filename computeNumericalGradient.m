function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

% Compute numerical gradient checking, and return the result in numgrad.
% The numerical gradient is computed from the mathematical definition of the
% derivative.

paramNum = length(theta);

EPSILON = 10^-4;
e = zeros(paramNum,1);

for i = 1:paramNum
    
    e(i) = 1;    
    numgrad(i) = (J(theta+EPSILON*e) - J(theta-EPSILON*e))/(2*EPSILON);
    e(i) = 0;
    
end

end
