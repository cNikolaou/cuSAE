/**
 * 
 *	This program computes the cost and the gradients of a sparse autoencoder
 *	neural network, to adjust its weights properly. It is a vectorized 
 *	implementation in CUDA C. It is can be called by a MATLAB(R) code file
 *	and will sufficiently compute the cost function and its gradient with
 *	respect to each weight variable.
 *
 *
 *	Author: Chistos Nikolaou
 *	Date: April-May 2014
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "computation_functions.h"
#include "mex.h"
#include "matrix.h"

#define IND(i,j,ld) (((j)*(ld))+(i))

// global variables
const int blocksize = 512;

// Declare functions that are defined in this source file
void SetInputVars(int thetaLength, int numberOfExamples, int features,
				  double *theta, double *data);
void SetHostMatrices(int visibleSize, int hiddenSize, double *theta,
					 double *hostW1, double *hostW2, 
					 double *hostb1, double *hostb2);
void TestInputMatValues(int visibleSize, int hiddenSize, 
						double *W1, double *W2, double *b1, double *b2);
void SetDeviceMatrices(int visibleSize, int hiddenSize,
				double *hostW1, double *hostW2, double *hostb1, double *hostb2, 
				double *W1, double *W2, double *b1, double *b2);
void SetGradVec(int visibleSize, int hiddenSize, 
				double *hostW1grad, double *hostW2grad, 
				double *hostb1grad, double *hostb2grad,
				double *gradVec);

/* --- SUBJECT TO CHANGE; TESTING STAGE --- */	
void squareMatrix(double *mat, int m, int n);
__global__ void squareElement(double *mat, int size);
void rowSum(cublasHandle_t handle, double *mat, int m, int n, double *sumMat);
/* --- END SUBJECT TO CHANGE AREA --- */


// main() function that is called by MATLAB(R) code
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	double *testTheta;

	testTheta = mxGetPr(prhs[0]); 
		
	double *matTheta, *matData;
	double matLambda, matSparsityParam, matBeta; 
	int matVisibleSize, matHiddenSize;

	int N = mxGetM(prhs[0]);
	
	// save inputs from MATLAB code
	matTheta = mxGetPr(prhs[0]);
	matVisibleSize = (int)mxGetScalar(prhs[1]);
	matHiddenSize = (int)mxGetScalar(prhs[2]);
	matLambda = (double)mxGetScalar(prhs[3]);
	matSparsityParam = (double)mxGetScalar(prhs[4]);
	matBeta = (double)mxGetScalar(prhs[5]);
	matData = mxGetPr(prhs[6]);

	// set CUDA variables
	cudaError_t cudaStat;
	cublasStatus_t cublasStat;
	cublasHandle_t handle;

	cublasCreate(&handle);


	// These are inputs to the MATLAB code
	double *theta, *data;
	theta = matTheta;
	data = matData;

	// declare train variables
	double lambda = matLambda;
	double sparsityParam = matSparsityParam;
	double beta = matBeta;		
	int visibleSize, hiddenSize;
	int numberOfExamples = mxGetN(prhs[6]);

	visibleSize = matVisibleSize;
	hiddenSize = matHiddenSize;

	int thetaLength = (2*visibleSize*hiddenSize + hiddenSize + visibleSize);

	// declare output variables - matrices
	double *matCost, *matGradVec;

	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(thetaLength, 1, mxREAL);

	matCost = (double*)mxGetPr(plhs[0]);
//	matGradVec = (double*)mxGetPr(plhs[1]);	

//	matGradVec = (double*) malloc(thetaLength*sizeof(*double));

	// allocate space for theta vector
//	theta = (double *) malloc(thetaLength * sizeof(*theta));

	// allocate host memory for 
//	data = (double *) malloc(numberOfExamples * visibleSize * sizeof(double));

	// print algorithm's information
/*	printf("Visible size = %d, ", visibleSize);
	printf("hidden size = %d, ", hiddenSize);
	printf("lambda = %f, ", lambda);
	printf("beta = %f, ", beta); 
	printf("sparsityParam = %f, ", sparsityParam);
	printf("thetaLength = %d\n", thetaLength);
*/

	// set inputs for testing if there are no input variables (and the source
	// file is not called by MATLAB(R) code)
//	SetInputVars(thetaLength, numberOfExamples, visibleSize, theta, data);

	int i,j;

	// print elements of theta vector
/*
	for(i = 0; i < thetaLength; i++) {
		printf("theta_double[%d] = %f \n", i , testTheta[i]);
	}

	printf("\n");
*/

	// print elements of theta vector
/*
	printf("Matrix theta:\n");
	for(i = 0; i < thetaLength; i++) {
			printf("theta[%d] = %2.2f \n", i, theta[i]);
	}
	printf("\n");
*/

	// print elements of the Data matrix
/*
	printf("DATA matrix\n");
	for(i = 0; i < visibleSize; i++) {
		for(j = 0; j < numberOfExamples; j++) {
			printf("dat[%d,%d]=%f ", i, j, data[IND(i,j,visibleSize)]);
		}
		printf("\n");
	}
	printf("\n");
*/


	/* ------------------------------------------------------------ */		
	/* ----- Set host (weight) matrices from the theta vector ----- */
	/* ------------------------------------------------------------ */		

	double *hostW1, *hostW2, *hostb1, *hostb2;
	hostW1 = (double*) malloc(hiddenSize*visibleSize*sizeof(double));
	hostW2 = (double*) malloc(visibleSize*hiddenSize*sizeof(double));
	hostb1 = (double*) malloc(hiddenSize*sizeof(double));
	hostb2 = (double*) malloc(visibleSize*sizeof(double));

	// Define host matrices with the right elements from
	// the theta matrix
	SetHostMatrices(visibleSize, hiddenSize, 
					theta, hostW1, hostW2, hostb1, hostb2);


	/* ------------------------------------- */	
	/* ----- Matrix transfer to device ----- */
	/* ------------------------------------- */	

	// Define device matrices
	double *W1, *W2, *b1, *b2;

	// Memory space for W1 matrix
	cudaStat = cudaMalloc((void**)&W1, visibleSize*hiddenSize*sizeof(double));
	if(cudaStat != cudaSuccess) {
		printf("Unable to malloc memory on device for W1.\n");
		exit(1);
	}

	// Memory space for W2 matrix
	cudaStat = cudaMalloc((void**)&W2, visibleSize*hiddenSize*sizeof(double));
	if(cudaStat != cudaSuccess) {
		printf("Unable to malloc memory on device for W2.\n");
		exit(1);
	}

	// Memory space for b1 matrix (vector)
	cudaStat = cudaMalloc((void**)&b1, hiddenSize*sizeof(double));
	if(cudaStat != cudaSuccess) {
		printf("Unable to malloc memory on device for b1.\n");
		exit(1);
	}

	// Memory space for b2 matrix (vector)
	cudaStat = cudaMalloc((void**)&b2, visibleSize*sizeof(double));
	if(cudaStat != cudaSuccess) {
		printf("Unable to malloc memory on device for b2.\n");
		exit(1);
	}

	SetDeviceMatrices(visibleSize, hiddenSize, 
					  hostW1, hostW2, hostb1, hostb2, W1, W2, b1, b2);


	/* --------------------------------------------------- */
	/* ----- Define host matrices to test the values ----- */
	/* --------------------------------------------------- */

	// define host matrices to test values that are
	// saved into the cuda memry space
//	TestInputMatValues(visibleSize, hiddenSize, W1, W2, b1, b2);


	/* ------------------------ */
	/* ----- Main program ----- */
	/* ------------------------ */
		
	// Device memory allocation for the layer output matrices
	double *y, *x, *a1, *z2, *a2, *z3, *a3;

	cudaStat = cudaMalloc((void**)&y, 
						  visibleSize*numberOfExamples*sizeof(double));
	cudaStat = cudaMalloc((void**)&x, 
						  visibleSize*numberOfExamples*sizeof(double));
	cudaStat = cudaMalloc((void**)&a1, 
						  visibleSize*numberOfExamples*sizeof(double));
	cudaStat = cudaMalloc((void**)&z2, 
						  hiddenSize*numberOfExamples*sizeof(double));
	cudaStat = cudaMalloc((void**)&a2,
						  hiddenSize*numberOfExamples*sizeof(double));
	cudaStat = cudaMalloc((void**)&z3, 
						  visibleSize*numberOfExamples*sizeof(double));
	cudaStat = cudaMalloc((void**)&a3, 
						  visibleSize*numberOfExamples*sizeof(double));


	/* ------------------------------- */
	/* ----- Forward Propagation ----- */
	/* ------------------------------- */

	// variables for the CUBLAS functions
	double a = 1.0;
	double b = 1.0;

	// set input to be equal to data
	cublasStat = cublasSetMatrix(visibleSize, numberOfExamples, sizeof(double),
								 data, visibleSize, x, visibleSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to set x equal to input data.\n");
		exit(1);
	}

	// set target output y to be equal to inpute x.
	cublasStat = cublasSetMatrix(visibleSize, numberOfExamples, sizeof(double),
								 data, visibleSize, y, visibleSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to set y equal to input data (autoencoder).\n");
		exit(1);
	}

	// set z2 to repetition of b1 and compute 
	// z2 = W1*a1 + repmat(b1,1,numberOfExamples)
	SetRepMat(hostb1, hiddenSize, numberOfExamples, z2);

	// x equals a1
	cublasStat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hiddenSize, 
							 numberOfExamples, visibleSize, &a, W1, hiddenSize,
							 x, visibleSize, &b, z2, hiddenSize); 
		
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to compute z2 = W1*a1 + z2 "); 
		printf("(=repmat(b1,1,numberOfExamples)).\n");
	}

	ComputeSigmoid(z2,hiddenSize*numberOfExamples,a2);

	// set z3 to repetition of b2 and compute 
	// z3 = W2*a2 + repmat(b2,1,numberOfExamples)
	SetRepMat(hostb2, visibleSize, numberOfExamples, z3);

	cublasStat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, visibleSize, 
							 numberOfExamples, hiddenSize, &a, W2, visibleSize, 
							 a2, hiddenSize, &b, z3, visibleSize);
				
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to compute z3 = W2*a2 + z3 ");
		printf("=repmap(b2,1,numberOfExamples)).\n");
	}

	ComputeSigmoid(z3,visibleSize*numberOfExamples,a3);


	/* ------------------------ */
	/* --- Back Propagation --- */
	/* ------------------------ */

	// Compute partial cost
	double *partCost, *delta3, *delta2;

	cudaStat = cudaMalloc((void**)&partCost, 
						  visibleSize*numberOfExamples*sizeof(double));
	cudaStat = cudaMalloc((void**)&delta2, 
						  hiddenSize*numberOfExamples*sizeof(double));
	cudaStat = cudaMalloc((void**)&delta3, 
						  visibleSize*numberOfExamples*sizeof(double));

	ComputePartCost(handle,a3,y,visibleSize,numberOfExamples,partCost);

	int gridsize = 1;

	// Comput delta
	dim3 d3Block(blocksize, 1);
	gridsize = (int) (visibleSize*numberOfExamples/blocksize + 1);
	dim3 dimGrid(gridsize, 1);
	
	// Print information that might be usefull
/*
	printf("Create block with %d threads: visibleSize*numberOfExamples\n", 
												visibleSize*numberOfExamples);
*/
	CompDelta3<<<dimGrid,d3Block>>>(y,a3,visibleSize*numberOfExamples,delta3);

	CompDelta(handle,W2,a2, hiddenSize,numberOfExamples,visibleSize,
			  delta3,delta2);


	/* ----------------------------------- */
	/* ----- Compute Error Gradients ----- */
	/* ----------------------------------- */
	
	// Device memory allocation for the derivatives of weight matrices
	double *DW1, *DW2, *Db1, *Db2;

	cudaStat = cudaMalloc((void**)&DW1, hiddenSize*visibleSize*sizeof(double));
	cudaStat = cudaMalloc((void**)&Db1, hiddenSize*sizeof(double));
	cudaStat = cudaMalloc((void**)&DW2, visibleSize*hiddenSize*sizeof(double));
	cudaStat = cudaMalloc((void**)&Db2, visibleSize*sizeof(double));


	// define variable for CUBLAS function call
	b = 0.0;

	// compute DW1 = delta2*a1'
	cublasStat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, hiddenSize, 
							 visibleSize, numberOfExamples,	&a, delta2, 
							 hiddenSize, x, visibleSize, &b, DW1, hiddenSize);

	// compute DW2 = delta3*a2'
	cublasStat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, visibleSize, 
							 hiddenSize, numberOfExamples, &a, delta3, 
							 visibleSize, a2, hiddenSize, &b, DW2, visibleSize);


	// temporary matrix to compute sum of delta
	double *onesVec;

	// compute Db1 = sum(delta2,2)
	cudaStat = cudaMalloc((void**)&onesVec, numberOfExamples*sizeof(double));

	dim3 onesBlock1(blocksize, 1);
	gridsize = (int) (numberOfExamples/blocksize + 1);
	dim3 onesGrid1(gridsize,1);

	// print information for debugging
/*	
	printf("Create block with %d threads: numberOfExamples\n", 
												numberOfExamples);
*/
	SetOnes<<<onesGrid1, onesBlock1>>>(numberOfExamples,onesVec);

	// define variable used for CUBLAS functions
	b = 0.0;

	cublasStat = cublasDgemv(handle, CUBLAS_OP_N, hiddenSize, 
							 numberOfExamples, &a, delta2, hiddenSize, 
							 onesVec, 1, &b, Db1, 1);

	// compute Db2 = sum(delta3,2) 

	b = 0.0;

	cublasStat = cublasDgemv(handle, CUBLAS_OP_N, visibleSize, 
							 numberOfExamples, &a, delta3, visibleSize, 
							 onesVec, 1, &b, Db2, 1);

	cudaFree(onesVec);


	/* ------------------------ */
	/* ----- Compute Cost ----- */
	/* ------------------------ */

	// define (device and host) cost matrices
	double cost, *hostCost, *tempCost;

	// allocate the appropriate space
	cudaStat = cudaMalloc((void**)&tempCost, sizeof(double));
	hostCost = (double*) malloc(sizeof(double));

	// compute sum(partCost)
	cudaStat = cudaMalloc((void**)&onesVec, numberOfExamples*sizeof(double));

	dim3 onesBlock3(blocksize,1);
	gridsize = (int) (numberOfExamples/blocksize + 1);
	dim3 onesGrid3(gridsize,1);
	SetOnes<<<onesGrid3,onesBlock3>>>(numberOfExamples,onesVec);

	b = 0.0;
		
	cublasStat = cublasDgemv(handle, CUBLAS_OP_T, numberOfExamples, 1,
							 &a, partCost, numberOfExamples, onesVec, 1, 
							 &b, tempCost, 1);

	cudaStat = cudaMemcpy(hostCost, tempCost, sizeof(double), 
						  cudaMemcpyDeviceToHost);


	/* ------------------------ */
	/* ----- Compute Cost ----- */
	/* ------------------------ */

	// Compute the square of the W1 and W2 weight matrices 
	double *sqrW1, *sqrW2;

	cudaStat = cudaMalloc((void**)&sqrW1, 
							hiddenSize*inputSize*sizeof(double));
	cudaStat = cudaMalloc((void**)&sqrW2, 
							inputSize*hiddenSize*sizeof(double));
	
	squareMatrix(sqrW1, hiddenSize, inputSize);
	squareMatrix(sqrW2, inputSize, hiddenSize);


	// Compute the row-wise sum of the (squared) W1 and W2 matrices
	double *rowSumW1, *rowSumW2; 

	cudaStat = cudaMalloc((void**)&rowSumW1, hiddenSize*sizeof(double));
	cudaStat = cudaMalloc((void**)&rowSumW2, visibleSize*sizeof(double));

	rowSum(handle, sqrW1, hiddenSize, visibleSize, rowSumW1);
	rowSum(handle, sqrW2, visibleSize, hiddenSize, rowSumW2);


	// Find total sum
	double *totSumW1, *totSumW2;

	cudaStat = cudaMalloc((void**)&totSumW1, sizeof(double));
	cudaStat = cudaMalloc((void**)&totSumW2, sizeof(double));

	colSum(handle, rowSumW1, hiddenSize, 1, totSumW1);
	colSum(handle, rowSumW2, visibleSize, 1, totSumW2);

	// Host variables that will hold the partial sums of the matrices
	double *partW1Cost, *partW2Cost;

	// allocate memeory space
	partW1Cost = malloc(sizeof(double));
	partW2Cost = malloc(sizeof(double));
	
	// copy valu for dvice to host
	cudaStat = cudaMemcpy(partW1Cost, totSumW1, sizeof(double), cudaMemcpyDeviceToHost);
	cudaStat = cudaMemcpy(partW2Cost, totSumW2, sizeof(double), cudaMemcpyDeviceToHost);

	if (cudaStat != cudaSuccess) {
		printf("Error while copying the total W1.^2 sum to host.");
		exit(1);
	}

	cost = 1/(double)numberOfExamples * (*hostCost); 
		// + lambda/2 * ((*partW1Cost) + (*partW2Cost));



	/* ----------------------------- */
	/* ----- Compute gradients ----- */
	/* ----------------------------- */

	// Host matrices that will take the DW values
	double *hostDW1, *hostDW2, *hostDb1, *hostDb2;

	hostDW1 = (double*) malloc(hiddenSize*visibleSize*sizeof(double));
	hostDW2 = (double*) malloc(visibleSize*hiddenSize*sizeof(double));
	hostDb1 = (double*) malloc(visibleSize*sizeof(double));
	hostDb2 = (double*) malloc(visibleSize*sizeof(double));


	// Get DW matrices from device matrices
	
	cublasStat = cublasGetMatrix(hiddenSize, visibleSize, sizeof(double), 
								 DW1, hiddenSize, hostDW1, hiddenSize);

	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR;"); 
		printf("Failed to copy DW1 device matrix to hostW1grad host matrix.\n");
		exit(1);
	}

	cublasStat = cublasGetMatrix(visibleSize, hiddenSize, sizeof(double), 
								 DW2, visibleSize, hostDW2, visibleSize);

	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR;"); 
		printf("Failed to copy DW1 device matrix to hostW1grad host matrix.\n");
		exit(1);
	}

	cublasStat = cublasGetMatrix(hiddenSize, 1, sizeof(double), 
								 Db1, hiddenSize, hostDb1, hiddenSize);

	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR;");
		printf("Failed to copy DW1 device matrix to hostW1grad host matrix.\n");
		exit(1);
	}

	cublasStat = cublasGetMatrix(visibleSize, 1, sizeof(double), 
								 Db2, visibleSize, hostDb2, visibleSize);

	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR;");
		printf("Failed to copy DW1 device matrix to hostW1grad host matrix.\n");
		exit(1);
	}

	
	// Compute the final values of the weight gradients

	// Host matrices that will hold the final gradient values
	double *hostW1grad, *hostW2grad, *hostb1grad, *hostb2grad;

	hostW1grad = (double*) malloc(hiddenSize*visibleSize*sizeof(double));
	hostW2grad = (double*) malloc(visibleSize*hiddenSize*sizeof(double));
	hostb1grad = (double*) malloc(hiddenSize*sizeof(double));
	hostb2grad = (double*) malloc(visibleSize*sizeof(double));


	// Set gradient final values
	CompWgrad(hostDW1, hiddenSize, visibleSize, numberOfExamples, 
			  lambda, hostW1, hostW1grad);
	CompWgrad(hostDW2, visibleSize, hiddenSize, numberOfExamples, 
			  lambda, hostW2, hostW2grad);
	Compbgrad(hostDb1, hiddenSize, numberOfExamples, hostb1grad);
	Compbgrad(hostDb2, visibleSize, numberOfExamples, hostb2grad);


	/* --------------------------------------------------- */
	/* ----- Define the gradient vector (theta grad) ----- */
	/* --------------------------------------------------- */

	double *gradVec;

	gradVec = (double*) malloc(thetaLength*sizeof(double));


	SetGradVec(visibleSize, hiddenSize, 
			   hostW1grad, hostW2grad, hostb1grad, hostb2grad,
			   gradVec);


	/* ---------------------------------------------- */
	/* ----- Print computed matrices for testing----- */
	/* ---------------------------------------------- */
/*
	printf("\nPrint matrix z2:\n");
	PrintReturnedMat(hiddenSize, numberOfExamples, z2);

	printf("\nPrint matrix a2:\n");
	PrintReturnedMat(hiddenSize, numberOfExamples, a2);

	printf("\nPrint matrix z3:\n");
	PrintReturnedMat(visibleSize, numberOfExamples, z3);

	printf("\nPrint matrix a3:\n");
	PrintReturnedMat(visibleSize, numberOfExamples, a3);

	printf("\nPrint matrix partCost:\n");
	PrintReturnedMat(numberOfExamples, 1, partCost);

	printf("\nPrint matrix delta3:\n");
	PrintReturnedMat(visibleSize, numberOfExamples, delta3);

	printf("\nPrint matrix delta2:\n");
	PrintReturnedMat(hiddenSize, numberOfExamples, delta2);

	printf("\nPrint matrix DW1:\n");
	PrintReturnedMat(hiddenSize, visibleSize, DW1);
		
	printf("\nPrint matrix DW2:\n");
	PrintReturnedMat(visibleSize, hiddenSize, DW2);

	printf("\nPrint matrix Db1:\n");
	PrintReturnedMat(hiddenSize, 1, Db1);

	printf("\nPrint matrix Db2:\n");
	PrintReturnedMat(visibleSize, 1, Db2);
		
	printf("\nPrint matrix tempCost:\n");
	PrintReturnedMat(1, 1, tempCost);

	printf("\nTotal cost is %f\n", cost);

*/

	/* ------------------------------ */
	/* ----- Print grad vectort ----- */
	/* ------------------------------ */


/*
	printf("\nTheta grad vector\n");
	printf("---------------------\n");

	matGradVec = mxGetPr(plhs[1]);
		
	for (i = 0; i < thetaLength; i++) {
		printf("i = %d : %f\n", i+1, gradVec[i]);
	}
*/

	
	for (i = 0; i < thetaLength; i++) {
		matGradVec[i] = gradVec[i];
	}
	

	*matCost = cost;
    //matGradVec = gradVec;
	
//	matGradVec = (double*)mxGetPr(plhs[1]);


	/* --------------------------------- */
	/* ----- Free allocated memory ----- */
	/* --------------------------------- */

	cublasDestroy(handle);
	
	cudaFree(W1); cudaFree(W2); cudaFree(b1); cudaFree(b2);
	cudaFree(DW1); cudaFree(DW2); cudaFree(Db1); cudaFree(Db2);
	cudaFree(y); cudaFree(x); cudaFree(a1); cudaFree(z2); cudaFree(a2);
	cudaFree(z3); cudaFree(a3);

	cudaFree(partCost); cudaFree(delta2); cudaFree(delta3);
}


void SetInputVars(int thetaLength, int numberOfExamples, int features,
				  double *theta, double *data) {

	for (int i = 0; i < thetaLength; i++) {
		if (i < 100) 
			theta[i] = 0.01*i;
		else
			theta[i] = 0.99;
	}

	for (int i = 0; i < features; i++) {
		for (int j = 0; j < numberOfExamples; j++) {
			data[IND(i,j,features)] = 0.5;
			//printf("%d %d %d\n", i, j, IND(i,j,features));
		}
	}
}


void SetHostMatrices(int visibleSize, int hiddenSize, double *theta,
					 double *hostW1, double *hostW2, 
					 double *hostb1, double *hostb2) {

	int offset = 0;

	//printf("\nTo hostW1:\n");
	
	for (int i = 0; i < hiddenSize; i++) {
		for (int j = 0; j < visibleSize; j++) {
			hostW1[IND(i,j,hiddenSize)] = theta[i*visibleSize+j];
	//		printf("%d = %f \n", IND(i,j,hiddenSize), 
	//				theta[i*visibleSize+j]);
		}
	}
	
	offset += hiddenSize*visibleSize;

	
	//printf("\nTo hostW2:\n");

	for (int i = 0; i < visibleSize; i++) {
		for (int j = 0; j < hiddenSize; j++) {
			hostW2[IND(i,j,visibleSize)] = theta[offset + i*hiddenSize+j];
	//		printf("%d = %f \n", IND(i,j,visibleSize), 
	//				theta[offset + i*hiddenSize+j]);
		}
	}
	
	offset += hiddenSize*visibleSize;

	
	//printf("\nTo hostb1:\n");

	for (int i = 0; i < hiddenSize; i++) {
		for (int j = 0; j < 1; j++) {
			hostb1[IND(i,j,hiddenSize)] = theta[offset +  i + visibleSize*j];
	//		printf("%d = %f \n", IND(i,j,hiddenSize), 
	//				theta[offset + i + visibleSize*j]);
		}
	}
	
	offset += hiddenSize;

	
	//printf("\nTo hostb2:\n");
	
	for (int i = 0; i < visibleSize; i++) {
		for (int j = 0; j < 1; j++) {
			hostb2[IND(i,j,visibleSize)] = theta[offset + i + hiddenSize*j];
	//		printf("%d = %f \n" , IND(i,j,hiddenSize), 
	//				theta[offset + i + hiddenSize*j]);
		}
	}

	offset += visibleSize;
	//printf("\nOffset is %d\n", offset);
}


void TestInputMatValues(int visibleSize, int hiddenSize, 
						double *W1, double *W2, double *b1, double *b2) {
	
	cublasStatus_t cublasStat;
	double *hostMat;

	/* --- Print W1 matrix --- */

	// host memory space allocation fot the W1 matrix
	hostMat = (double *) malloc(visibleSize*hiddenSize*sizeof(double));

	// get elements for W1 matrix
	cublasStat = cublasGetMatrix(hiddenSize, visibleSize, sizeof(double), 
								 W1, hiddenSize, hostMat, hiddenSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to get matrix W1.\n");
		exit(1);
	}

	// print W1 elements
	printf("Matrix W1:\n");
	for (int i = 0; i < hiddenSize; i++) {
		for (int j = 0; j < visibleSize; j++) {
			printf("W1[%d,%d] = %2.2f, ", i, j, hostMat[IND(i,j,hiddenSize)]);
		}
		printf("\n");
	}
	printf("\n");


	/* --- Print W2 matrix --- */
	
	// host memory space allocation for the W2 matrix
	hostMat = (double *) malloc(visibleSize*hiddenSize*sizeof(double));

	// get elements for W2 matrix
	cublasStat = cublasGetMatrix(hiddenSize, visibleSize, sizeof(double), 
								 W2, hiddenSize, hostMat, hiddenSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to get matrix W2.\n");
		exit(1);
	}

	// print W2 elements
	printf("Matrix W2:\n");
	for (int i = 0; i < visibleSize; i++) {
		for (int j = 0; j < hiddenSize; j++) {
			printf("W2[%d,%d] = %2.2f, ", i, j, hostMat[i*hiddenSize+j]);
		}
		printf("\n");
	}
	printf("\n");


	/* --- Print b1 matrix --- */
	
	// host memory allocation foe the b1 matrix (vector)
	hostMat = (double *) malloc(hiddenSize*sizeof(double));

	// get elements fpr b2 matrix
	cublasStat = cublasGetMatrix(hiddenSize, 1, sizeof(double), 
								 b1, hiddenSize, hostMat, hiddenSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to get matrix b1.\n");
		exit(1);
	}

	// printf b1 elements
	printf("Matrix b1:\n");
	for (int i = 0; i < hiddenSize; i++) {
		printf("b1[%d] = %2.2f\n", i, hostMat[i]);
	}
	printf("\n");


	/* --- Print b2 matrix --- */

	// host memory allocation for the b2 matrix (vector)
	hostMat = (double *) malloc(visibleSize*sizeof(double));

	// get elements for b2 matrix
	cublasStat = cublasGetMatrix(visibleSize, 1, sizeof(double), 
								 b2, visibleSize, hostMat, visibleSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to get matrix b2.\n");
		exit(1);
	}

	// print b2 elements
	printf("Matrix b2:\n");
	for (int i = 0; i < visibleSize; i++) {
		printf("b2[%d] = %2.2f\n", i, hostMat[i]);
	}
	printf("\n");

	free(hostMat);
}


void SetDeviceMatrices(int visibleSize, int hiddenSize,
				double *hostW1, double *hostW2, double *hostb1, double *hostb2, 
				double *W1, double *W2, double *b1, double *b2) {

	cublasStatus_t cublasStat;		

	// Set W1 device matrix
	cublasStat = cublasSetMatrix(hiddenSize, visibleSize, sizeof(double), 
						     	 hostW1, hiddenSize, W1, hiddenSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to create matrix W1.\n");
		exit(1);
	}

	// Set W2 device matrix
	cublasStat = cublasSetMatrix(visibleSize, hiddenSize, sizeof(double), 
								 hostW2, visibleSize, W2, visibleSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to create matrix W2.\n");
		exit(1);
	}
	
	// Set b1 device matrix (vector)
	cublasStat = cublasSetMatrix(hiddenSize, 1, sizeof(double), 
								 hostb1, hiddenSize, b1, hiddenSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to create matrix b1.\n");
		exit(1);
	}

	// Set b2 device matrix (vector)
	cublasStat = cublasSetMatrix(visibleSize, 1, sizeof(double), 
								 hostb2, visibleSize, b2, visibleSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to create matrix b2.\n");
		exit(1);
	}

}


void SetGradVec(int visibleSize, int hiddenSize, 
				double *hostW1grad, double *hostW2grad, 
				double *hostb1grad, double *hostb2grad,
				double *gradVec) {


	int offset = 0;

	//printf("\nFrom hostW1grad:\n");
	
	for (int i = 0; i < hiddenSize; i++) {
		for (int j = 0; j < visibleSize; j++) {
			gradVec[i*visibleSize+j] = hostW1grad[i+j*hiddenSize]; 
//			printf("position %d , place %f \n",	i*visibleSize+j,
//				  	hostW1grad[i*visibleSize+j]);
		}
	}

	offset += hiddenSize*visibleSize;

	//printf("\nFrom hostW2grad:\n");

	for (int i = 0; i < visibleSize; i++) {
		for (int j = 0; j < hiddenSize; j++) {
			gradVec[offset + i*hiddenSize + j] = hostW2grad[i+j*visibleSize];
//			printf("position %d , place %f \n", offset + i*hiddenSize + j, 
//				   	hostW2grad[IND(i,j,visibleSize)]);
		}
	}

	offset += hiddenSize*visibleSize;

	
	//printf("\nFrom hostb1grad:\n");
	
	for (int i = 0; i < hiddenSize; i++) {
		for (int j = 0; j < 1; j++) {
			gradVec[offset + i + visibleSize*j] = hostb1grad[i];
//			printf("position %d , place %f \n",	offset + i + visibleSize*j, 
//				   	hostb1grad[IND(i,j,hiddenSize)]);
		}
	}

	offset += hiddenSize;

	
	//printf("\nFrom hostb2grad:\n");

	for (int i = 0; i < visibleSize; i++) {
		for (int j = 0; j < 1; j++) {
			gradVec[offset + i + hiddenSize*j] = hostb2grad[i];
//			printf("position %d , place %f \n", offset + i + hiddenSize*j, 
//				   	hostb2grad[IND(i,j,visibleSize)]);
		}
	}

	offset += visibleSize;
	//printf("\nOffset is %d\n", offset);
}

/* --- SUBJECT TO CHANGE; TESTING STAGE --- */	
void squareMatrix(double *mat, int m, int n) {

	int numberOfElements = m*n;

	dim3 sqrBlock(blocksize,1);
	int gridsize = (int) (numberOfElements/blocksize + 1);
	dim3 sqrGrid(gridsize,1);
	squareElement<<<sqrGrid, sqrBlock>>>(mat, numberOfElements);

};

__global__ void squareElement(double *mat, int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numberOfElements)
		mat[index] = mat[index]*mat[index];

}

void rowSum(cublasHandle_t handle, double *mat, int m, int n, double *sum) {

	cudaError_t cudaStat;
	cublasStatus_t cublasStat;

	double *onesVec;

	cudaStat = cudaMalloc((void**)&onesVec, n*sizeof(double));

	if (cudaStat != cudaSuccess) {
		printf("Error while allocation device space\n");
		printf("for onesVec in rowSum function.\n");
		exir(1);
	}
	
	dim3 onesBlock(blocksize,1);
	int gridsize = (int) n/blocksize;
	dim3 onesGrid(gridsize,1);
	SetOnes<<<onesGrid, onesBlock>>>(n,onesVec);

	double a = 1.0;
	double b = 0.0;

	culasStat = cublasDgemv(handle, CUBLAS_OP_N, m, n, 
							&a, mat, n, onesVec, 1, 
							&b, sum, 1);

	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS ERROR; \n");
		printf("Unbale to compute row-wise sum of the matrix\n");
		exit(1);
	}
}

void rowSum(cublasHandle_t, double *mat, int m, int n, double *sum) {

	cudaError_t cudaStat;
	cublasStatus_t cublasStat;

	double *onesVec;

	cudaStat = cudaMalloc((void**)&onesVec, m*sizeof(double));

	if (cudaStat != cudaSuccess) {
		printf("Error while allocation device space\n");
		printf("for onesVec in rowSum function.\n");
		exir(1);
	}

	dim3 onesBlock(blocksize,1);
	int gridsize = (int) m/blocksize;
	dim3 onesGrid(gridsize,1);
	SetOnes<<<onesGrid, onesBlock>>>(onesVec);

	double a = 1.0;
	double b = 0.0;

	cublasStat = cublasDgemv(handle, CUBLAS_OP_T, m, n, 
							 &a, mat, n, onesVec, 1, 
							 &b, sum ,1);
	
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS ERROR; \n");
		printf("Unbale to compute row-wise sum of the matrix\n");
		exit(1);
	}
}
/* --- END SUBJECT TO CHANGE AREA --- */
