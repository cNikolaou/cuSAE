/**
 * 
 *	This program computes the cost and the gradients of a sparse autoencoder
 *	neural network, to adjust its weights properly. It is a vectorized 
 *	implementation in CUDA C. It is a prototype and is only used to test
 *	the CUDA algorithm with a small set of artificial examples (artificial 
 *	dataset and weights).
 * 
 *
 *	compile it with nvcc -lcublas sparseAutoencoderCost.cu
 *
 *
 *	Author: Chistos Nikolaou
 *	Date: April 2014
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "computation_functions.h"

#define IND(i,j,ld) (((j)*(ld))+(i))

// Define functions
void SetInputVars(float *theta, float *data, int thetaLength, 
				  int numberOfExamples, int features);
void SetHostMatrices(int visibleSize, int hiddenSize, float *theta,
					 float *hostW1, float *hostW2, 
					 float *hostb1, float *hostb2);
void TestInputMatValues(int visibleSize, int hiddenSize, 
						float *W1, float *W2, float *b1, float *b2);
void SetDeviceMatrices(int visibleSize, int hiddenSize,
				float *hostW1, float *hostW2, float *hostb1, float *hostb2, 
				float *W1, float *W2, float *b1, float *b2);
void SetGradVec(int visibleSize, int hiddenSize, float *gradVec, 
				float *hostW1grad, float *hostW2grad, 
				float *hostb1grad, float *hostb2grad);

int main(void) {

	// set CUDA variables
	cudaError_t cudaStat;
	cublasStatus_t cublasStat;
	cublasHandle_t handle;

	cublasCreate(&handle);

	// These are inputs to the MATLAB code
	float *theta, *data;
	float lambda = 1;
	float sparsityParam = 0.1;
	float beta = 1;
	int visibleSize, hiddenSize;
	int numberOfExamples = 3;

	visibleSize = 10;
	hiddenSize = 4;

	// Define matrices
	float *W1, *W2, *b1, *b2;

	// allocate space for theta vector
	int thetaLength = (2*visibleSize*hiddenSize + hiddenSize + visibleSize);
	theta = (float *) malloc(thetaLength * sizeof(*theta));

	// allocate host memory for 
	data = (float *) malloc(numberOfExamples*visibleSize*sizeof(float));

	// print algorithm's information
	printf("Visible size = %d, ", visibleSize);
	printf("hidden size = %d, ", hiddenSize);
	printf("lambda = %f, ", lambda);
	printf("beta = %f, ", beta); 
	printf("sparsityParam = %f, ", sparsityParam);
	printf("thetaLength = %d\n", thetaLength);

	// set inputs for testing
	SetInputVars(theta, data, thetaLength, numberOfExamples, visibleSize);

	int i,j;

	printf("\n");
	printf("Matrix theta:\n");
	for(i = 0; i < thetaLength; i++) {
			printf("theta[%d] = %2.2f \n", i, theta[i]);
	}
	printf("\n");

	printf("DATA matrix\n");
	for(i = 0; i < visibleSize; i++) {
		for(j = 0; j < numberOfExamples; j++) {
			printf("dat[%d,%d]=%f ", i, j, data[IND(i,j,visibleSize)]);
		}
		printf("\n");
	}
	printf("\n");

	
	/* ----- Set host (weight) matrices from the theta vector ----- */
	float *hostW1, *hostW2, *hostb1, *hostb2;
	hostW1 = (float*) malloc(hiddenSize*visibleSize*sizeof(float));
	hostW2 = (float*) malloc(visibleSize*hiddenSize*sizeof(float));
	hostb1 = (float*) malloc(hiddenSize*sizeof(float));
	hostb2 = (float*) malloc(visibleSize*sizeof(float));

	SetHostMatrices(visibleSize, hiddenSize, 
					theta, hostW1, hostW2, hostb1, hostb2);

	
	/* ----- Matrix transfer to device ----- */

	// Memory space for W1 matrix
	cudaStat = cudaMalloc((void**)&W1, visibleSize*hiddenSize*sizeof(float));
	if(cudaStat != cudaSuccess) {
		printf("Unable to malloc memory on device for W1.\n");
		exit(1);
	}

	// Memory space for W2 matrix
 	cudaStat = cudaMalloc((void**)&W2, visibleSize*hiddenSize*sizeof(float));
	if(cudaStat != cudaSuccess) {
		printf("Unable to malloc memory on device for W2.\n");
		exit(1);
	}

	// Memory space for b1 matrix (vector)
	cudaStat = cudaMalloc((void**)&b1, hiddenSize*sizeof(float));
	if(cudaStat != cudaSuccess) {
		printf("Unable to malloc memory on device for b1.\n");
		exit(1);
	}

	// Memory space for b2 matrix (vector)
	cudaStat = cudaMalloc((void**)&b2, visibleSize*sizeof(float));
	if(cudaStat != cudaSuccess) {
		printf("Unable to malloc memory on device for b2.\n");
		exit(1);
	}

	SetDeviceMatrices(visibleSize, hiddenSize, 
					  hostW1, hostW2, hostb1, hostb2, W1, W2, b1, b2);


	/* ----- Define host matrices to test the values ----- */

	TestInputMatValues(visibleSize, hiddenSize, W1, W2, b1, b2);


	/* ----- Main program ----- */
	
	// Device memory allocation for the layer output matrices
	float *y, *x, *a1, *z2, *a2, *z3, *a3;

	cudaStat = cudaMalloc((void**)&y, 
						  visibleSize*numberOfExamples*sizeof(float));
	cudaStat = cudaMalloc((void**)&x, 
						  visibleSize*numberOfExamples*sizeof(float));
	cudaStat = cudaMalloc((void**)&a1, 
						  visibleSize*numberOfExamples*sizeof(float));
	cudaStat = cudaMalloc((void**)&z2, 
						  hiddenSize*numberOfExamples*sizeof(float));
	cudaStat = cudaMalloc((void**)&a2,
						  hiddenSize*numberOfExamples*sizeof(float));
	cudaStat = cudaMalloc((void**)&z3, 
						  visibleSize*numberOfExamples*sizeof(float));
	cudaStat = cudaMalloc((void**)&a3, 
						  visibleSize*numberOfExamples*sizeof(float));


	/* ----- Forward Propagation ----- */

	float a = 1.0;
	float b = 1.0;

	// set input to be equal to data
	cublasStat = cublasSetMatrix(visibleSize, numberOfExamples, sizeof(float),
								 data, visibleSize, x, visibleSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to set x equal to input data.\n");
		exit(1);
	}

	// set target output y to be equal to inpute x.
	cublasStat = cublasSetMatrix(visibleSize, numberOfExamples, sizeof(float),
								 data, visibleSize, y, visibleSize);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to set y equal to input data (autoencoder).\n");
		exit(1);
	}

	// set z2 to repetition of b1 and compute 
	// z2 = W1*a1 + repmat(b1,1,numberOfExamples)
	SetRepMat(z2, hostb1, hiddenSize, numberOfExamples);

	// x equals a1
	cublasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hiddenSize, 
							 numberOfExamples, visibleSize, &a, W1, hiddenSize,
							 x, visibleSize, &b, z2, hiddenSize); 
	
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to compute z2 = W1*a1 + z2 "); 
		printf("(=repmat(b1,1,numberOfExamples)).\n");
	}

	ComputeSigmoid(z2,a2,hiddenSize*numberOfExamples);

	// set z3 to repetition of b2 and compute 
	// z3 = W2*a2 + repmat(b2,1,numberOfExamples)
	SetRepMat(z3, hostb2, visibleSize, numberOfExamples);

	cublasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, visibleSize, 
							 numberOfExamples, hiddenSize, &a, W2, visibleSize, 
							 a2, hiddenSize, &b, z3, visibleSize);
			
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to compute z3 = W2*a2 + z3 ");
		printf("=repmap(b2,1,numberOfExamples)).\n");
	}

	ComputeSigmoid(z3,a3,visibleSize*numberOfExamples);


	/* --- Back Propagation ---*/

	// Parital Cost
	float *partCost, *delta3, *delta2;

	cudaStat = cudaMalloc((void**)&partCost, 
						  visibleSize*numberOfExamples*sizeof(float));
	cudaStat = cudaMalloc((void**)&delta2, 
						  hiddenSize*numberOfExamples*sizeof(float));
	cudaStat = cudaMalloc((void**)&delta3, 
						  visibleSize*numberOfExamples*sizeof(float));

	ComputePartCost(handle,a3,y,partCost,visibleSize,numberOfExamples);

	// Delta
	dim3 d3Block(visibleSize*numberOfExamples);
	dim3 dimGrid(1,1);
	printf("Create block with %d threads: visibleSize*numberOfExamples\n", 
												visibleSize*numberOfExamples);

	CompDelta3<<<dimGrid,d3Block>>>(y,a3,delta3,visibleSize*numberOfExamples);

	CompDelta(handle,W2,delta3,a2,delta2,
			  hiddenSize,numberOfExamples,visibleSize);



	/* ----- Compute Error Gradients ----- */

	// Device memory allocation for the derivatives of weight matrices
	float *DW1, *DW2, *Db1, *Db2;

	cudaStat = cudaMalloc((void**)&DW1, hiddenSize*visibleSize*sizeof(float));
	cudaStat = cudaMalloc((void**)&Db1, hiddenSize*sizeof(float));
	cudaStat = cudaMalloc((void**)&DW2, visibleSize*hiddenSize*sizeof(float));
	cudaStat = cudaMalloc((void**)&Db2, visibleSize*sizeof(float));


	b = 0.0;

	// compute DW1 = delta2*a1'
	cublasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, hiddenSize, 
							 visibleSize, numberOfExamples,	&a, delta2, 
							 hiddenSize, x, visibleSize, &b, DW1, hiddenSize);

	// compute DW2 = delta3*a2'
	cublasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, visibleSize, 
							 hiddenSize, numberOfExamples, &a, delta3, 
							 visibleSize, a2, hiddenSize, &b, DW2, visibleSize);


	float *onesVec;

	// compute Db1 = sum(delta2,2)
	cudaStat = cudaMalloc((void**)&onesVec, numberOfExamples*sizeof(float));

	dim3 onesBlock1(numberOfExamples,1);
	dim3 onesGrid1(1,1);
	printf("Create block with %d threads: numberOfExamples\n", 
												numberOfExamples);
	SetOnes<<<onesGrid1, onesBlock1>>>(onesVec,numberOfExamples);

	b = 0.0;

	cublasStat = cublasSgemv(handle, CUBLAS_OP_N, hiddenSize, 
							 numberOfExamples, &a, delta2, hiddenSize, 
							 onesVec, 1, &b, Db1, 1);

	// compute Db2 = sum(delta3,2) 

	b = 0.0;

	cublasStat = cublasSgemv(handle, CUBLAS_OP_N, visibleSize, 
							 numberOfExamples, &a, delta3, visibleSize, 
							 onesVec, 1, &b, Db2, 1);

	cudaFree(onesVec);


	/* ----- Compute Cost ----- */

	float cost, *hostCost, *tempCost;

	cudaStat = cudaMalloc((void**)&tempCost, sizeof(float));
	hostCost = (float*) malloc(sizeof(float));

	cudaStat = cudaMalloc((void**)&onesVec, numberOfExamples*sizeof(float));

	dim3 onesBlock3(numberOfExamples,1);
	dim3 onesGrid3(1,1);
	SetOnes<<<onesGrid3,onesBlock3>>>(onesVec, numberOfExamples);

	b = 0.0;
	
	cublasStat = cublasSgemv(handle, CUBLAS_OP_T, numberOfExamples, 1,
							 &a, partCost, numberOfExamples, onesVec, 1, 
							 &b, tempCost, 1);

	cudaStat = cudaMemcpy(hostCost, tempCost, sizeof(float), cudaMemcpyDeviceToHost);

	cost = 1/(float)numberOfExamples * (*hostCost);



	/* ----- Compute gradients ----- */

	float *hostW1grad, *hostW2grad, *hostb1grad, *hostb2grad;

	hostW1grad = (float*) malloc(hiddenSize*visibleSize*sizeof(float));
	hostW2grad = (float*) malloc(visibleSize*hiddenSize*sizeof(float));
	hostb1grad = (float*) malloc(hiddenSize*sizeof(float));
	hostb2grad = (float*) malloc(visibleSize*sizeof(float));


	cublasStat = cublasGetMatrix(hiddenSize, visibleSize, sizeof(float), 
								 DW1, hiddenSize, hostW1grad, hiddenSize);

	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR; Failed to copy DW1 device matrix to hostW1grad host matrix.\n");
		exit(1);
	}

	cublasStat = cublasGetMatrix(visibleSize, hiddenSize, sizeof(float), 
								 DW2, visibleSize, hostW2grad, visibleSize);

	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR; Failed to copy DW1 device matrix to hostW1grad host matrix.\n");
		exit(1);
	}

	cublasStat = cublasGetMatrix(hiddenSize, 1, sizeof(float), 
								 Db1, hiddenSize, hostb1grad, hiddenSize);

	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR; Failed to copy DW1 device matrix to hostW1grad host matrix.\n");
		exit(1);
	}

	cublasStat = cublasGetMatrix(visibleSize, 1, sizeof(float), 
								 Db2, visibleSize, hostb2grad, visibleSize);

	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR; Failed to copy DW1 device matrix to hostW1grad host matrix.\n");
		exit(1);
	}


	// Set grafient final values
	CompWgrad(hostW1grad, hiddenSize, visibleSize, numberOfExamples);
	CompWgrad(hostW2grad, visibleSize, hiddenSize, numberOfExamples);
	Compbgrad(hostb1grad, hiddenSize, numberOfExamples);
	Compbgrad(hostb2grad, visibleSize, numberOfExamples);

	/* ----- Define the gradient vector (theta grad) ----- */


	float *gradVec;

	gradVec = (float*) malloc(thetaLength*sizeof(float));


	SetGradVec(visibleSize, hiddenSize, gradVec, 
			   hostW1grad, hostW2grad, hostb1grad, hostb2grad);



	/* ----- Print computed matrices for testing----- */
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


	/* ----- Print grad vectort -----*/


	printf("\nTheta grad vector\n");
	printf("---------------------\n");
	for (i = 0; i < thetaLength; i++)
//		printf("i = %d : %f\n", i+1, gradVec[i]);



	/* ----- Free allocated memory ----- */
	cublasDestroy(handle);
	
	cudaFree(W1); cudaFree(W2); cudaFree(b1); cudaFree(b2);
	cudaFree(DW1); cudaFree(DW2); cudaFree(Db1); cudaFree(Db2);
	cudaFree(y); cudaFree(x); cudaFree(a1); cudaFree(z2); cudaFree(a2);
	cudaFree(z3); cudaFree(a3);

	cudaFree(partCost); cudaFree(delta2); cudaFree(delta3);
}


void SetInputVars(float *theta, float *data, 
				  int thetaLength, int numberOfExamples, int features) {

	int i, j;

	for(i = 0; i < thetaLength; i++) {
		if(i < 100) 
			theta[i] = 0.01*i;
		else
			theta[i] = 0.99;
	}

	for(i = 0; i < features; i++) {
		for(j = 0; j < numberOfExamples; j++) {
			data[IND(i,j,features)] = 0.5;
			//printf("%d %d %d\n", i, j, IND(i,j,features));
		}
	}
}


void SetHostMatrices(int visibleSize, int hiddenSize, float *theta,
					 float *hostW1, float *hostW2, 
					 float *hostb1, float *hostb2) {

	int i,j;

	int offset = 0;

	printf("\nTo hostW1:\n");
	
	for(i = 0; i < hiddenSize; i++) {
		for(j = 0; j < visibleSize; j++) {
			hostW1[IND(i,j,hiddenSize)] = theta[i*visibleSize+j];
			printf("%d = %f \n", IND(i,j,hiddenSize), 
					theta[i*visibleSize+j]);
		}
	}
	
	offset += hiddenSize*visibleSize;

	
	printf("\nTo hostW2:\n");

	for(i = 0; i < visibleSize; i++) {
		for(j = 0; j < hiddenSize; j++) {
			hostW2[IND(i,j,visibleSize)] = theta[offset + i*hiddenSize+j];
			printf("%d = %f \n", IND(i,j,visibleSize), 
					theta[offset + i*hiddenSize+j]);
		}
	}
	
	offset += hiddenSize*visibleSize;

	
	printf("\nTo hostb1:\n");

	for(i = 0; i < hiddenSize; i++) {
		for(j = 0; j < 1; j++) {
			hostb1[IND(i,j,hiddenSize)] = theta[offset +  i + visibleSize*j];
			printf("%d = %f \n", IND(i,j,hiddenSize), 
					theta[offset + i + visibleSize*j]);
		}
	}
	
	offset += hiddenSize;

	
	printf("\nTo hostb2:\n");
	
	for(i = 0; i < visibleSize; i++) {
		for(j = 0; j < 1; j++) {
			hostb2[IND(i,j,visibleSize)] = theta[offset + i + hiddenSize*j];
			printf("%d = %f \n" , IND(i,j,hiddenSize), 
					theta[offset + i + hiddenSize*j]);
		}
	}

	offset += visibleSize;
	printf("\nOffset is %d\n", offset);
}


void TestInputMatValues(int visibleSize, int hiddenSize, 
						float *W1, float *W2, float *b1, float *b2) {
	
	cublasStatus_t cublasStat;
	float *hostMat;
	int i,j;

	/* --- Print W1 matrix --- */

	// host memory space allocation fot the W1 matrix
	hostMat = (float *) malloc(visibleSize*hiddenSize*sizeof(float));

	// get elements for W1 matrix
	cublasStat = cublasGetMatrix(hiddenSize, visibleSize, sizeof(float), 
								 W1, hiddenSize, hostMat, hiddenSize);
	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to get matrix W1.\n");
		exit(1);
	}

	// print W1 elements
	printf("Matrix W1:\n");
	for(i = 0; i < hiddenSize; i++) {
		for(j = 0; j < visibleSize; j++) {
			printf("W1[%d,%d] = %2.2f, ", i, j, hostMat[IND(i,j,hiddenSize)]);
		}
		printf("\n");
	}
	printf("\n");


	/* --- Print W2 matrix --- */
	
	// host memory space allocation for the W2 matrix
	hostMat = (float *) malloc(visibleSize*hiddenSize*sizeof(float));

	// get elements for W2 matrix
	cublasStat = cublasGetMatrix(hiddenSize, visibleSize, sizeof(float), 
								 W2, hiddenSize, hostMat, hiddenSize);
	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to get matrix W2.\n");
		exit(1);
	}

	// print W2 elements
	printf("Matrix W2:\n");
	for(i = 0; i < visibleSize; i++) {
		for(j = 0; j < hiddenSize; j++) {
			printf("W2[%d,%d] = %2.2f, ", i, j, hostMat[i*hiddenSize+j]);
		}
		printf("\n");
	}
	printf("\n");


	/* --- Print b1 matrix --- */
	
	// host memory allocation foe the b1 matrix (vector)
	hostMat = (float *) malloc(hiddenSize*sizeof(float));

	// get elements fpr b2 matrix
	cublasStat = cublasGetMatrix(hiddenSize, 1, sizeof(float), 
								 b1, hiddenSize, hostMat, hiddenSize);
	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to get matrix b1.\n");
		exit(1);
	}

	// printf b1 elements
	printf("Matrix b1:\n");
	for(i = 0; i < hiddenSize; i++) {
		printf("b1[%d] = %2.2f\n", i, hostMat[i]);
	}
	printf("\n");


	/* --- Print b2 matrix --- */

	// host memory allocation for the b2 matrix (vector)
	hostMat = (float *) malloc(visibleSize*sizeof(float));

	// get elements for b2 matrix
	cublasStat = cublasGetMatrix(visibleSize, 1, sizeof(float), 
								 b2, visibleSize, hostMat, visibleSize);
	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to get matrix b2.\n");
		exit(1);
	}

	// print b2 elements
	printf("Matrix b2:\n");
	for(i = 0; i < visibleSize; i++) {
		printf("b2[%d] = %2.2f\n", i, hostMat[i]);
	}
	printf("\n");

	free(hostMat);
}


void SetDeviceMatrices(int visibleSize, int hiddenSize,
				float *hostW1, float *hostW2, float *hostb1, float *hostb2, 
				float *W1, float *W2, float *b1, float *b2) {

	cublasStatus_t cublasStat;		

	// Set W1 device matrix
	cublasStat = cublasSetMatrix(hiddenSize, visibleSize, sizeof(float), 
						     	 hostW1, hiddenSize, W1, hiddenSize);
	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to create matrix W1.\n");
		exit(1);
	}

	// Set W2 device matrix
	cublasStat = cublasSetMatrix(visibleSize, hiddenSize, sizeof(float), 
								 hostW2, visibleSize, W2, visibleSize);
	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to create matrix W2.\n");
		exit(1);
	}
	
	// Set b1 device matrix (vector)
	cublasStat = cublasSetMatrix(hiddenSize, 1, sizeof(float), 
								 hostb1, hiddenSize, b1, hiddenSize);
	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to create matrix b1.\n");
		exit(1);
	}

	// Set b2 device matrix (vector)
	cublasStat = cublasSetMatrix(visibleSize, 1, sizeof(float), 
								 hostb2, visibleSize, b2, visibleSize);
	if(cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to create matrix b2.\n");
		exit(1);
	}

}


void SetGradVec(int visibleSize, int hiddenSize, float *gradVec, 
				float *hostW1grad, float *hostW2grad, 
				float *hostb1grad, float *hostb2grad) {

	int i,j;

	int offset = 0;

	printf("\nFrom hostW1grad:\n");
	
	for(i = 0; i < hiddenSize; i++) {
		for(j = 0; j < visibleSize; j++) {
			gradVec[i*visibleSize+j] = hostW1grad[i*visibleSize+j]; 
			printf("position %d , place %f \n",	i*visibleSize+j,
				  	hostW1grad[i*visibleSize+j]);
		}
	}

	offset += hiddenSize*visibleSize;

	printf("\nFrom hostW2grad:\n");

	for(i = 0; i < visibleSize; i++) {
		for(j = 0; j < hiddenSize; j++) {
			gradVec[offset + i*hiddenSize + j] = hostW2grad[i*hiddenSize+j];
			printf("position %d , place %f \n", offset + i*hiddenSize + j, 
				   	hostW2grad[IND(i,j,visibleSize)]);
		}
	}

	offset += hiddenSize*visibleSize;

	
	printf("\nFrom hostb1grad:\n");
	
	for(i = 0; i < hiddenSize; i++) {
		for(j = 0; j < 1; j++) {
			gradVec[offset + i + visibleSize*j] = hostb1grad[i];
			printf("position %d , place %f \n",	offset + i + visibleSize*j, 
				   	hostb1grad[IND(i,j,hiddenSize)]);
		}
	}

	offset += hiddenSize;

	
	printf("\nFrom hostb2grad:\n");

	for(i = 0; i < visibleSize; i++) {
		for(j = 0; j < 1; j++) {
			gradVec[offset + i + hiddenSize*j] = hostb2grad[i];
			printf("position %d , place %f \n", offset + i + hiddenSize*j, 
				   	hostb2grad[IND(i,j,visibleSize)]);
		}
	}

	offset += visibleSize;
	printf("\nOffset is %d\n", offset);
}
