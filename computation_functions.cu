/**
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

// global variables
const int blocksize = 512;

__global__ void Sigmoid(const double *a, const int numberOfElements, 
						double *sa) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numberOfElements)
		sa[index] = 1/(1+expf(-a[index]));

}

__global__ void ComputeAbsDiff(const double *hx, const double *y, int N, 
							   double *diff) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < N)
		diff[index] = pow(abs(hx[index]-y[index]),2);
}



__global__ void CompDelta3(const double *y, const double *a3, int length, 
						   double *delta3) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < length)
		delta3[index] = -(y[index]-a3[index])*a3[index]*(1-a3[index]);
}

__global__ void CompDelta(const double *a2, int N, double *delta) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < N)
		delta[index] = delta[index] * a2[index] * (1-a2[index]); 
}

__global__ void SetOnes(int length, double *ones) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < length)
		ones[index] = 1.0;

}

__global__ void SetZeros(int length, double *zeros) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < length)
		zeros[index] = 0.0;

}

void ComputeSigmoid(const double *z, int N, double *a) {

	// set a2 = sigmoid(z2)
	dim3 dimBlock(blocksize,1);
	int gridsize = (int) (N/blocksize + 1);
	dim3 dimGrid(gridsize,1);
	//printf("Create block with %d threads : N in computeSigmoid\n", N);
	Sigmoid<<<dimGrid, dimBlock>>>(z, N, a);

}
void SetRepMat(const double *b, int numberOfRows, int numberOfCols, double *z) {
	
//	printf("Size: %d, %d\n", numberOfRows, numberOfCols);
	double *temp;
	temp = (double *) malloc(numberOfRows*numberOfCols*sizeof(double));

	for (int i = 0; i < numberOfRows; i++)  {
		for (int j = 0; j < numberOfCols; j++) {
			temp[IND(i,j,numberOfRows)] = b[i]; 
//			printf("%d,%d \n", i, j);
		}
	}

	if (cublasSetMatrix(numberOfRows, numberOfCols, sizeof(double), 
			temp, numberOfRows, z, numberOfRows) != CUBLAS_STATUS_SUCCESS) {

		printf("ERROR; Cannot set repetition matrix z this time.\n");
		exit(1);
	}

	free(temp);
}
void ComputePartCost(cublasHandle_t handle, const double *hx, const double *y, 
					 int numberOfRows, int numberOfCols, double *partCost) {

	/* --- Compute the squared absolute difference of the two matrices --- */

	int N = numberOfRows*numberOfCols;

	double *diff;
	cudaMalloc((void**)&diff, numberOfRows*numberOfCols*sizeof(double));
	
	dim3 dimBlock(blocksize,1);
	int gridsize = (int) (N/blocksize + 1);
	dim3 dimGrid(gridsize,1);
	//printf("Create block with %d threads : N (in computePartCost)\n", N);
	ComputeAbsDiff<<<dimGrid, dimBlock>>>(hx, y, N, diff);
	
//	printf("\nPrint matrix abs(hx-y).^2\n");
//	PrintReturnedMat(numberOfRows, numberOfCols, diff);


	/* --- Define a vector with ones --- */

	double *onesVec;
	cudaMalloc((void**)&onesVec, numberOfRows*sizeof(double));

	dim3 onesBlock(blocksize, 1);
	//printf("Create block with %d threads : numberOfRows (computePartCost)\n"
	//		, numberOfRows);
	gridsize = (int) (numberOfRows/blocksize + 1);
	dim3 onesGrid(gridsize, 1);
	SetOnes<<<onesGrid, onesBlock>>>(numberOfRows,onesVec);

//	printReturnedMat(numberOfRows, 1, onesVec);


	/* --- Compute the sum of each column of the diff matrix ---*/

	double a = 0.5;
	double b = 0.0;
	
//	dim3 zerosBlock(numberOfCols,1);
//	setZeros<<<dimGrid, zerosBlock>>>(partCost, numberOfCols);

	cublasSgemv(handle, CUBLAS_OP_T, numberOfRows, numberOfCols, 
				&a, diff, numberOfRows, onesVec, 1, &b,
				partCost, 1);	


	// Free temporary matrices
	cudaFree(diff);
	cudaFree(onesVec);
}



void CompDelta(cublasHandle_t handle, const double *W2, const double *a2, 
			   int hiddenSize, int numberOfExamples, int visibleSize, 
			   double *delta3, double *delta2) {

	// compute W2'*delta3	
	double a = 1.0;
	double b = 0.0;

	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, hiddenSize, numberOfExamples, 
				visibleSize, &a, W2, visibleSize, delta3, visibleSize, &b, 
				delta2, hiddenSize);

	int N = hiddenSize * numberOfExamples;

	dim3 dimBlock(blocksize ,1);
	int gridsize = (int) (N/blocksize + 1);
	dim3 dimGrid(gridsize, 1);
	//printf("Create block with %d threads : N in compDelta\n", N);
	CompDelta<<<dimGrid,dimBlock>>>(a2, N, delta2);

}

void CompWgrad(double *W, int numberOfRows, int numberOfCols, int m) {

	int i,j;

	for (i = 0; i < numberOfRows; i++) {
		for (j = 0; j < numberOfCols; j++) {
			W[i*numberOfCols + j] = 1/(double)m * W[i*numberOfCols + j];
		}
	}
}

void Compbgrad(double *b, int numberOfRows, int m) {

	int i;

	for(i = 0; i < numberOfRows; i++) {
		b[i] = 1/(double)m * b[i];
	}
}

void PrintReturnedMat(int numberOfRows, int numberOfCols, 
					  const double *deviceMat) {

	double *ret;
	ret = (double *) malloc(numberOfRows*numberOfCols*sizeof(double));

	if (cublasGetMatrix(numberOfRows, numberOfCols, sizeof(double), 
						deviceMat, numberOfRows, ret, numberOfRows) 
						!= CUBLAS_STATUS_SUCCESS) {

		printf("Cannot get matrix from the device for printing\n");
		exit(1);
	}

	printf("---------------------------\n");
	for (int i = 0; i < numberOfRows; i++) {
		for (int j = 0; j < numberOfCols; j++) {
			printf("RetMat[%d,%d] = %1.2f  ",
					i, j, ret[IND(i,j,numberOfRows)]);
		}
		printf("\n");
	}
	printf("WARNING: Values are rounded to two decimals\n");

	free(ret);
}


