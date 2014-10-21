/**
 * 
 *	This file contains a part of the computation functions that are needed to 
 *	sufficiently compute the cost function and it's grandient with respect to
 *	each network's weight variable. These functions are called from the main 
 *	(or equivalently the mexFunction() function) in the 
 *  sparseAutoencoderCost.cu file.
 *
 *
 *  Author: Chistos Nikolaou
 *  Date: April-May 2014
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

	if (index < length) {
		// the order of computations mught have an impact on the final outcome
    // the last term is a smaller number than the other two so it is 
    // multiplied after the first to (between 0 and 1) number are multiplied
    // so they are closer in scale
    delta3[index] = -a3[index]*(1-a3[index])*(y[index]-a3[index]);
  }
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
__global__ void squareElement(const double *mat, const int numberOfElements,
                              double *matSqr) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numberOfElements) {
		matSqr[index] = mat[index]*mat[index];
  }

}
void ComputeSigmoid(const double *z, int N, double *a) {

	// set a2 = sigmoid(z2)
	dim3 dimBlock(blocksize,1);
	int gridsize = (int) (N/blocksize + 1);
	dim3 dimGrid(gridsize,1);
	//printf("Create block with %d threads : N in computeSigmoid\n", N);
	Sigmoid<<<dimGrid, dimBlock>>>(z, N, a);

}
void SetRepMat(const double *b, const int numberOfRows, 
               const int numberOfCols, double *z) {
	
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

void DevRepMat(const cublasHandle_t handle,
               const double *b, const int numberOfRows,
               const int numberOfCols, double *z) {
  
  double alpha = 1.0;
  double beta = 0.0;

  double *onesVec;
  cudaMalloc((void**)&onesVec, numberOfCols*sizeof(double));
  
  dim3 onesBlock(blocksize, 1);
  int gridsize = (int) (blocksize/numberOfCols + 1);
  dim3 onesGrid(gridsize, 1);

  SetOnes<<<onesGrid, onesBlock>>>(numberOfCols, onesVec);

  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, numberOfRows, numberOfCols,
              1, &alpha, b, numberOfRows, onesVec, 1, &beta, z, numberOfRows);
  
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

  //CHECK THIS!!!!
	cublasDgemv(handle, CUBLAS_OP_T, numberOfRows, numberOfCols, 
				&a, diff, numberOfRows, onesVec, 1, &b,
				partCost, 1);	


	// Free temporary matrices
	cudaFree(diff);
	cudaFree(onesVec);
}

__global__ void CompDeltaSecondTerm(const double sparsityParam,
                                    const double *rho,
                                    const int N, const double beta,
                                    double *secondTerm) {

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < N) {
    secondTerm[idx] = beta*(-sparsityParam/rho[idx] +
                          (1 - sparsityParam)/(1 - rho[idx])); 
  }

}

__global__ void CompDelta2(const double sparsityParam,
                            const double *rho,
                            const int N, const double beta,
                            const double *a2,
                            double *delta2) {
  
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int tix = threadIdx.x;

  if (idx < N) {
    delta2[idx] = (beta*(-sparsityParam/rho[tix] + 
                  (1 - sparsityParam)/(1 - rho[tix])) +
                  delta2[idx]) * a2[idx] * (1 - a2[idx]);
  }

}

/*
void CompDelta(cublasHandle_t handle, const double *W2, const double *a2, 
			   int hiddenSize, int numberOfExamples, int visibleSize, 
         const double *rho, const double sparsityParam, 
         const double beta, const double *delta3, 
         double *delta2) {

  //repmat(beta*(-sparsityParam./rho + (1-sparsityParam)./(1-rho)),1,m)).*a2.*(1-a2); 
  
  // Compute the second term
  //-sparsityParam./rho + (1-sparsityParam)./(1-rho)

  double *temp;

  if (cudaSuccess != cudaMalloc((void**)&temp, hiddenSize*sizeof(double))) {
    printf("ERROR in allocating space for temp vector. In CompDelta.\n");
  }

  dim3 dimBlockSecondTerm(blocksize, 1); 
  int gridsize_2 = (int) (hiddenSize/blocksize + 1);
  dim3 dimGridSecondTerm(gridsize_2, 1);

  CompDeltaSecondTerm<<<dimGridSecondTerm, dimBlockSecondTerm>>>
                      (sparsityParam, rho, hiddenSize, beta, temp);
  
//  PrintReturnedMat(hiddenSize, 1, temp);

  DevRepMat(handle, temp, hiddenSize, numberOfExamples, delta2);

  cudaFree(temp);

//  PrintReturnedMat(hiddenSize, 10, delta2);
*/
/**/
	// compute W2'*delta3	
/*	double a = 1.0;
	double b = 1.0;

	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, hiddenSize, numberOfExamples, 
				visibleSize, &a, W2, visibleSize, delta3, visibleSize, &b, 
				delta2, hiddenSize);

	int N = hiddenSize * numberOfExamples;

	dim3 dimBlockFirstTerm(blocksize, 1);
	int gridsize_1 = (int) (N/blocksize + 1);
	dim3 dimGridFirstTerm(gridsize_1, 1);
	//printf("Create block with %d threads : N in compDelta\n", N);
	CompDelta<<<dimGridFirstTerm,dimBlockFirstTerm>>>(a2, N, delta2);


}
*/

void CompDelta(cublasHandle_t handle, const double *W2, const double *a2, 
			   int hiddenSize, int numberOfExamples, int visibleSize, 
         const double *rho, const double sparsityParam, 
         const double beta, const double *delta3, 
         double *delta2) {

  double a = 1.0;
	double b = 0.0;

  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, hiddenSize, numberOfExamples, 
				visibleSize, &a, W2, visibleSize, delta3, visibleSize, &b, 
				delta2, hiddenSize);
  
  dim3 delta2Block(hiddenSize, 1);
  int gridsize = numberOfExamples;
  dim3 delta2Grid(gridsize, 1);

  // no need to use it; just in case something changes
  int N_2 = numberOfExamples*hiddenSize;

  CompDelta2<<<delta2Grid, delta2Block>>>(sparsityParam, rho, N_2,
                                          beta, a2, delta2);

}


void CompWgrad(const double *DW, const int numberOfRows, 
               const int numberOfCols, const int m, const double lambda, 
               const double *W, double *Wgrad) {

//  printf("In CompWgrad, lambda = %0.4f\n", lambda);
  
	int i,j;

	for (i = 0; i < numberOfRows; i++) {
		for (j = 0; j < numberOfCols; j++) {
      Wgrad[IND(i,j,numberOfRows)] = 1/(double)m * DW[IND(i,j,numberOfRows)] +
										lambda * W[IND(i,j,numberOfRows)];
/*      // For testing pursposes
      printf("%2.3f = %2.3f + %2.3f \n", 
            Wgrad[IND(i,j,numberOfRows)], 
            1/(double)m * DW[IND(i,j,numberOfRows)], 
            lambda * W[IND(i,j,numberOfRows)]);
*/

//			Wgrad[i*numberOfCols + j] = 1/(double)m * DW[i*numberOfCols + j] +
//										lambda * W[i*numberOfCols + j];
		}
	}
}

void Compbgrad(const double *Db, const int numberOfRows, const int m, 
               double *bgrad) {

	int i;

	for(i = 0; i < numberOfRows; i++) {
		bgrad[i] = 1/(double)m * Db[i];
	}

}

__global__ void CompPartKL(const double sparsityParam, const double *rho,
                           const int rho_size, double *temp) {
  
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < rho_size) {
    temp[idx] = sparsityParam*log(sparsityParam/rho[idx]) +
                (1 - sparsityParam)*log((1 - sparsityParam)/(1 - rho[idx]));
  }
  
}


void CompKL(const cublasHandle_t handle,
            const double sparsityParam, const double *rho, 
            const int rho_size, double *kl) {

  double *temp, *deviceKL;
  
  cudaMalloc((void**)&temp, rho_size*sizeof(double)); 
  cudaMalloc((void**)&deviceKL, sizeof(double));

  dim3 KLblock(blocksize, 1);
  int gridsize = (int) (blocksize/rho_size + 1);
  dim3 KLgrid(gridsize, 1);

  CompPartKL<<<KLgrid, KLblock>>>(sparsityParam, rho, rho_size, temp);
  
  ColSum(handle, temp, rho_size, 1, 1.0, deviceKL);

//  PrintReturnedMat(1, 1, deviceKL);

  cudaMemcpy(kl, deviceKL, sizeof(double), cudaMemcpyDeviceToHost);
  
  cudaFree(temp); cudaFree(deviceKL);
  
}

void squareMatrix(const double *mat, const int m, const int n, 
                  double *matSqr) {

	int numberOfElements = m*n;

	dim3 sqrBlock(blocksize,1);
	int gridsize = (int) (numberOfElements/blocksize + 1);
	dim3 sqrGrid(gridsize,1);
	squareElement<<<sqrGrid, sqrBlock>>>(mat, numberOfElements, matSqr);

};

void RowSum(const cublasHandle_t handle, const double *mat, 
            const int m, const int n, const double scale, double *sum) {


  
  size_t freeCudaMem, totalCudaMem;
  
  if (cudaSuccess != cudaMemGetInfo(&freeCudaMem, &totalCudaMem)) {
    printf("ERROR while trying to get information about device's memory.\n");
  } else {
    printf("Device total memory: %zd \tDevice free memory: %zd\n",
           totalCudaMem, freeCudaMem);  
  }

  printf("Try to allocate: %d bytes\n", n);

	cudaError_t cudaStat;
	cublasStatus_t cublasStat;

	double *onesVec;

	cudaStat = cudaMalloc((void**)&onesVec, n*sizeof(double));

	if (cudaStat != cudaSuccess) {
		printf("Error while allocation device space\n");
		printf("for onesVec in rowSum function.\n");


    printf("Error is: %s\n", cudaGetErrorString(cudaStat));
  


		exit(1);  

	}
	
	dim3 onesBlock(blocksize,1);
	int gridsize = (int) n/blocksize + 1;
	dim3 onesGrid(gridsize,1);
	SetOnes<<<onesGrid, onesBlock>>>(n,onesVec);

	double a = scale;
	double b = 0.0;


	cublasStat = cublasDgemv(handle, CUBLAS_OP_N, m, n, 
							&a, mat, m, onesVec, 1, 
							&b, sum, 1);

	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS ERROR; \n");
		printf("Unbale to compute row-wise sum of the matrix\n");
		exit(1);
	}

  cudaFree(onesVec);
}

void ColSum(const cublasHandle_t handle, const double *mat, 
            const int m, const int n, const double scale, double *sum) {

	cudaError_t cudaStat;
	cublasStatus_t cublasStat;

	double *onesVec;

	cudaStat = cudaMalloc((void**)&onesVec, m*sizeof(double));

	if (cudaStat != cudaSuccess) {
		printf("Error while allocation device space\n");
		printf("for onesVec in colSum function.\n");
    printf("Error is: %s\n", cudaGetErrorString(cudaStat));

    size_t freeCudaMem, totalCudaMem;
  
    if (cudaSuccess != cudaMemGetInfo(&freeCudaMem, &totalCudaMem)) {
      printf("ERROR while trying to get information about device's memory.\n");
    } else {
      printf("Device total memory: %zd \tDevice free memory: %zd\n",
             totalCudaMem, freeCudaMem);  
    }

		exit(1);
	}

	dim3 onesBlock(blocksize,1);
	int gridsize = (int) m/blocksize + 1;
	dim3 onesGrid(gridsize,1);
	SetOnes<<<onesGrid, onesBlock>>>(m,onesVec);

	double a = scale;
	double b = 0.0;


	cublasStat = cublasDgemv(handle, CUBLAS_OP_T, m, n, 
							 &a, mat, m, onesVec, 1, 
							 &b, sum ,1);
	
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS ERROR; \n");
		printf("Unbale to compute row-wise sum of the matrix\n");
		exit(1);
	}

  
  cudaFree(onesVec);
}

void PrintHostMat(int numberOfRows, int numberOfCols,
                  const double *hostMat) {
  
	printf("---------------------------\n");
	for (int i = 0; i < numberOfRows; i++) {
		for (int j = 0; j < numberOfCols; j++) {
			printf("Ret[%d,%d] = %1.3f  ",
					i, j, hostMat[IND(i,j,numberOfRows)]);
		}
		printf("\n");
	}
	printf("WARNING: Values are rounded to two decimals\n");

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

  PrintHostMat(numberOfRows, numberOfCols, ret);

	free(ret);
}



