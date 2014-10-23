%%  Train.m
%
%  This file is the train.m file from the UFLDL tutorial. It is modified
%  by Christos Nikolaou for the cuSAE program, which has a CUDA C 
%  implementation of the sparseAutoencodeCost() function.
%

clc
clear all;
close all force;

% execute make through matlab
system(sprintf('make clean; make'));

%%======================================================================
%% STEP 0: Parameters values
%
patchsize = 28;
numpatches = 10000;
visibleSize = patchsize^2;   % number of input units 
hiddenSize = 196;			 % number of hidden units 
sparsityParam = 0.1;		 % desired average activation of the hidden units.

lambda = 3e-3;				% weight decay parameter       
beta = 3;					% weight of sparsity penalty term       
%}


%%======================================================================
%% STEP 1: Get images, using the sampleIMAGES function. 
%  
%  If Step 3 is needed (gradient checking), then it is needed to check the
%  gradient in a smaller network and with fewer examples. Uncomment the
%  appropriate lines of code 
%

figure(1);
% plot 200 randomly selected image patches


patches = sampleIMAGES(patchsize, numpatches);
display_network(patches(:,randi(size(patches,2),200,1)),8);

% Uncomment these lines if there is a need for gradient checking (Step 3)
% (faster version for gradient checking)
%{
visibleSize = 100;
patches = patches(1:visibleSize,1:100);
hiddenSize = 40;
%}


%  Obtain random parameters theta
%rng(0);
theta = initializeParameters(hiddenSize, visibleSize);

%%======================================================================
%% STEP 2: Check sparseAutoencoderCost function.
%
%  Feel free to change the training settings when debugging your
%  code.  (For example, reducing the training set size or 
%  number of hidden units may make your code run faster; and setting beta 
%  and/or lambda to zero may be helpful for debugging.)
%

[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, patches);


%disp(grad);
%disp([[1:length(grad)]' grad]);
%cost
%
%
%%======================================================================
%% STEP 3: Gradient Checking
%
% For debugging urposes, performing gradient checking on smaller models 
% and smaller training sets (e.g., using only 10 training examples and 1-2 
% hidden units) may speed things up.

%{

% Check whether the numerical gradient is correct for a simple function. 
checkNumericalGradient();


% HINT: Reduce the size of hidden layer and/or the patch size to make a
% fast checking of the gradient computation. Otherwise it will take more
% than 20 minutes.

% Check the cost function and derivative calculations for 
% the sparse autoencoder.  
numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
                                                  hiddenSize, lambda, ...
                                                  sparsityParam, beta, ...
                                                  patches), theta);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small; usually less than 1e-9.

%}                                   
%%======================================================================
%% STEP 4: Sparse Autoencoder Training
%
%  Start training the sparse autoencoder with minFunc (L-BFGS).
%
% Calculate time
tic
%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
%
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

toc
%}
%
%%======================================================================
%% STEP 5: Visualization 

figure(2);

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 
date = datestr(clock);

fname = ['weights-' date(1:6) '.jpg'];

print -djpeg weights.jpg   % save the visualization to a file 

%}
