#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "computation_functions.h"

#define IND(i,j,ld) (((j)*(ld))+(i))


__global__ void Sigmoid(const float *a, const int numberOfElements, 
						float *sa) {

	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < numberOfElements)
		sa[index] = 1/(1+expf(-a[index]));

}

__global__ void ComputeAbsDiff(const float *hx, const float *y, int N, 
							   float *diff) {
	
	int index = threadIdx.x;

	if (index < N)
		diff[index] = pow(abs(hx[index]-y[index]),2);
}



__global__ void CompDelta3(const float *y, const float *a3, int length, 
						   float *delta3) {

	int index = threadIdx.x;

	if (index < length)
		delta3[index] = -(y[index]-a3[index])*a3[index]*(1-a3[index]);
}

__global__ void CompDelta(const float *a2, int N, float *delta) {

	int index = threadIdx.x;

	if (index < N)
		delta[index] = delta[index] * a2[index] * (1-a2[index]); 
}

__global__ void SetOnes(int length, float *ones) {

	int index = threadIdx.x;

	if (index < length)
		ones[index] = 1.0;

}

__global__ void SetZeros(int length, float *zeros) {

	int index = threadIdx.x;

	if (index < length)
		zeros[index] = 0.0;

}

void ComputeSigmoid(const float *z, int N, float *a) {

	// set a2 = sigmoid(z2)
	dim3 dimBlock(N,1);
	dim3 dimGrid(1,1);
	printf("Create block with %d threads : N in computeSigmoid\n", N);
	Sigmoid<<<dimGrid, dimBlock>>>(z, N, a);

}
void SetRepMat(const float *b, int numberOfRows, int numberOfCols, float *z) {
	
//	printf("Size: %d, %d\n", numberOfRows, numberOfCols);
	float *temp;
	temp = (float *) malloc(numberOfRows*numberOfCols*sizeof(float));

	for (int i = 0; i < numberOfRows; i++)  {
		for (int j = 0; j < numberOfCols; j++) {
			temp[IND(i,j,numberOfRows)] = b[i]; 
//			printf("%d,%d \n", i, j);
		}
	}

	if (cublasSetMatrix(numberOfRows, numberOfCols, sizeof(float), 
			temp, numberOfRows, z, numberOfRows) != CUBLAS_STATUS_SUCCESS) {

		printf("ERROR; Cannot set repetition matrix z this time.\n");
		exit(1);
	}

	free(temp);
}
void ComputePartCost(cublasHandle_t handle, const float *hx, const float *y, 
					 int numberOfRows, int numberOfCols, float *partCost) {

	/* --- Compute the squared absolute difference of the two matrices --- */

	int N = numberOfRows*numberOfCols;

	float *diff;
	cudaMalloc((void**)&diff, numberOfRows*numberOfCols*sizeof(float));
	
	dim3 dimBlock(N,1);
	dim3 dimGrid(1,1);
	printf("Create block with %d threads : N (in computePartCost)\n", N);
	ComputeAbsDiff<<<dimGrid, dimBlock>>>(hx, y, N, diff);
	
//	printf("\nPrint matrix abs(hx-y).^2\n");
//	PrintReturnedMat(numberOfRows, numberOfCols, diff);


	/* --- Define a vector with ones --- */

	float *onesVec;
	cudaMalloc((void**)&onesVec, numberOfRows*sizeof(float));

	dim3 onesBlock(numberOfRows, 1);
	printf("Create block with %d threads : numberOfRows (computePartCost)\n"
			, numberOfRows);
	SetOnes<<<dimGrid, onesBlock>>>(numberOfRows,onesVec);

//	printReturnedMat(numberOfRows, 1, onesVec);


	/* --- Compute the sum of each column of the diff matrix ---*/

	float a = 0.5;
	float b = 0.0;
	
//	dim3 zerosBlock(numberOfCols,1);
//	setZeros<<<dimGrid, zerosBlock>>>(partCost, numberOfCols);

	cublasSgemv(handle, CUBLAS_OP_T, numberOfRows, numberOfCols, 
				&a, diff, numberOfRows, onesVec, 1, &b,
				partCost, 1);	


	// Free temporary matrices
	cudaFree(diff);
	cudaFree(onesVec);
}



void CompDelta(cublasHandle_t handle, const float *W2, const float *a2, 
			   int hiddenSize, int numberOfExamples, int visibleSize, 
			   float *delta3, float *delta2) {

	// compute W2'*delta3	
	float a = 1.0;
	float b = 0.0;

	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, hiddenSize, numberOfExamples, 
				visibleSize, &a, W2, visibleSize, delta3, visibleSize, &b, 
				delta2, hiddenSize);

	int N = hiddenSize * numberOfExamples;

	dim3 dimBlock(N ,1);
	dim3 dimGrid(1, 1);
	printf("Create block with %d threads : N in compDelta\n", N);
	CompDelta<<<dimGrid,dimBlock>>>(a2, N, delta2);

}

void CompWgrad(float *W, int numberOfRows, int numberOfCols, int m) {

	int i,j;

	for (i = 0; i < numberOfRows; i++) {
		for (j = 0; j < numberOfCols; j++) {
			W[i*numberOfCols + j] = 1/(float)m * W[i*numberOfCols + j];
		}
	}
}

void Compbgrad(float *b, int numberOfRows, int m) {

	int i;

	for(i = 0; i < numberOfRows; i++) {
		b[i] = 1/(float)m * b[i];
	}
}

void PrintReturnedMat(int numberOfRows, int numberOfCols, 
					  const float *deviceMat) {

	float *ret;
	ret = (float *) malloc(numberOfRows*numberOfCols*sizeof(float));

	if (cublasGetMatrix(numberOfRows, numberOfCols, sizeof(float), 
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


