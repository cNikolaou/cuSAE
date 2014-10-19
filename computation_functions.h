/**
 *
 *	This is the header file for the functions' declaration. More information
 *	on the computation_functions.h file.
 *
 *
 *	Author: Christos Nikolaou
 *	Date: April-May 2014
 *
 */

#ifndef CUSAE_SRC_COMPUTATION_FUNCTIONS_H
#define CUSAE_SRC_COMPUTATION_FUNCTIONS_H

__global__ void Sigmoid(const double *a, const int numberOfElements, 
						double *sa);
__global__ void ComputeAbsDiff(const double *hx, const double *y, int N, 
							   double *diff);
__global__ void CompDelta3(const double *y, const double *a3, int length, 
						   double *delta3);
__global__ void CompDelta(const double *a2, int N, double *delta);
__global__ void SetOnes(int length, double *ones);
__global__ void SetZeros(int length, double *zeros);
__global__ void squareElement(const double *mat, const int size,
                  double *matSqr);

void ComputeSigmoid(const double *z, int N, double *a);
void SetRepMat(const double *b, int numberOfRows, int numberOfCols, double *z);
void ComputePartCost(cublasHandle_t handle, const double *hx, const double *y, 
					 int numberOfRows, int numberOfCols, double *partCost);
void CompDelta(cublasHandle_t handle, const double *W2, const double *a2, 
			   int hiddenSize, int numberOfExamples, int visibleSize, 
			   double *delta3, double *delta2);
void CompWgrad(const double *DW, const int numberOfRows, 
               const int numberOfCols, const int m, const double lambda, 
               const double *W, double *Wgrad);
void Compbgrad(const double *Db, const int numberOfRows, const int m, 
               double *bgrad);
void squareMatrix(const double *mat, const int m, const int n, 
                  double *matSqr);
void rowSum(const cublasHandle_t handle, const double *mat, 
            const int m, const int n, double *sum);
void colSum(const cublasHandle_t handle, const double *mat, 
            const int m, const int n, double *sum);
void PrintHostMat(int numberOfRows, int numberOfCols,
                  const double *hostMat);
void PrintReturnedMat(int numberOfRows, int numberOfCols, 
					            const double *deviceMat);

#endif
