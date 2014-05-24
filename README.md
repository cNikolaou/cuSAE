cuSAE
=====

A sparse autoencoder trained on MNIST images with MATLAB(R) and CUDA C. Cost 
and gradient function is implemented in CUDA C.

Heavily under development.

Code works on GNU/Linux 3.5.0-44-generic x86.


### Requirements:

1. MATLAB(R) R2013b (8.2.0.701)
2. gcc version 4.8.1
3. CUDA compilation tools, release 6.0
4. CUBLAS version 2
5. Training set: [train-images.idx3-ubyte] (http://yann.lecun.com/exdb/mnist/)


### Note:

minFunc subdirectory is a 3rd party software implementing L-BFGS optimization,
that is licensed under a Creative Commons, Attribute, Non-Commercial license. 
If you need to use this software for commercial purposes, you can download and
use a different function (fminlbfgs) that can serve the same purpose, but runs
~3x slower. The function fminlbfgs gives a BSD license implementation of the 
L-BFGS (so it is appropropriate to use it in commercial applications. More 
[information about Fminlbfgs] (http://www.mathworks.com/matlabcentral/fileexchange/23245-fminlbfgs--fast-limited-memory-optimizer) 
page.
