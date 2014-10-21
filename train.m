%%  Train.m
%
%  This file is the train.m file from the UFLDL tutorial. It is modified
%  by Christos Nikolaou for the cuSAE program.
%

clc
clear all;
close all force;

system(sprintf('make clean; make'));

tic;

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

% For testing pursposes of the CUDA code
%{
patchsize = 3;
numpatches = 5;
visibleSize = 10;
hiddenSize = 4;
sparsityParam = 0.1;
lambda = 3e-3;
beta = 3;
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


% For testing pursposes of the CUDA code
%patches = [0.05*[1:1:10]' 0.07*[1:1:10]' 0.09*[1:1:10]'];

patches = sampleIMAGES(patchsize, numpatches);
%display_network(patches(:,randi(size(patches,2),200,1)),8);


%
% Uncomment these lines if there is a need for gradient checking (Step 3)
% (faster version for gradient checking)
%
visibleSize = 250;
patches = patches(1:visibleSize,1:100);
hiddenSize = 4;
%}

% For Gradient checking (second choice - slower)
%{
patches = patches(:,1:1000);
hiddenSize = 2;
%}
%patches = patches(:,1:1000);


%  Obtain random parameters theta
rng(0);
theta = initializeParameters(hiddenSize, visibleSize);

% For testing pursposes
%theta = 1/(2*hiddenSize*visibleSize + hiddenSize + visibleSize) * ...
%          [1:2*hiddenSize*visibleSize + hiddenSize + visibleSize]';

%%======================================================================
%% STEP 2: Check sparseAutoencoderCost function.
%
%  Feel free to change the training settings when debugging your
%  code.  (For example, reducing the training set size or 
%  number of hidden units may make your code run faster; and setting beta 
%  and/or lambda to zero may be helpful for debugging.)
%

%display('Start cost');
[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, patches);


disp(grad);
%disp([[1:length(grad)]' grad]);
cost
%display('End cost');
%
%{
%%======================================================================
%% STEP 3: Gradient Checking
%
% Hint: If you are debugging your code, performing gradient checking on smaller models 
% and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
% units) may speed things up.

% First, lets make sure your numerical gradient computation is correct for a
% simple function.  After you have implemented computeNumericalGradient.m,
% run the following: 
checkNumericalGradient();

% Christos Nikolaou
% HINT: Reduce the size of hidden layer and/or the patch size to make a
% fast checking of the gradient computation. Otherwise it will take more
% than 20 minutes.

% Now we can use it to check your cost function and derivative calculations
% for the sparse autoencoder.  
numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
                                                  hiddenSize, lambda, ...
                                                  sparsityParam, beta, ...
                                                  patches), theta);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 
%}                                   
%%======================================================================
%% STEP 4: Sparse Autoencoder Training
%
%  Start training your sparse autoencoder with minFunc (L-BFGS).
%
%{
%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);
display('Before LBFGS');
%  Use minFunc to minimize the function
%
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

display('Ready for LBFGS');
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);
%}

%{
%  Testing code it may not be needed. 

[opttheta, cost] = minFunc(@(p) sparseAutoencoderCost(single(p), ...
				single(visibleSize), single(hiddenSize), ...
				single(lambda), single(sparsityParam), ...
				sigle(beta), single(patches)), ...
				theta, options);
%}

%{
%%======================================================================
%% STEP 5: Visualization 

figure(2);

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 
date = datestr(clock);

fname = ['weights-' date(1:6) '.jpg'];

print -djpeg weights.jpg   % save the visualization to a file 

%}
toc
