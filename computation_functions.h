#ifndef CUSAE_SRC_COMPUTATION_FUNCTIONS_H
#define CUSAE_SRC_COMPUTATION_FUNCTIONS_H

__global__ void Sigmoid(float *a, float *sa, int numberOfElements);
__global__ void ComputeAbsDiff(float *hx, float *y, float *diff, int N);
__global__ void CompDelta3(float *y, float *a3, float *delta3, int length);
__global__ void CompDelta(float *delta, float *a2, int N);
__global__ void SetOnes(float *ones, int length);
__global__ void SetZeros(float *zeros, int length);

void ComputeSigmoid(float *z, float *a, int N);
void SetRepMat(float *z, float *b, int numberOfRows, int numberOfCols);
void ComputePartCost(cublasHandle_t handle, float *hx, float *y, 
					 float *partCost, int numberOfRows, int numberOfCols);
void CompDelta(cublasHandle_t handle, float *W2, float *delta3, float *a2, 
					 float *delta2, int hiddenSize,	
					 int numberOfExamples, int visibleSize);
void CompWgrad(float *W, int numberOfRows, int numberOfCols, int m);
void Compbgrad(float *b, int numberOfRows, int m);
void PrintReturnedMat(int numberOfRows, int numberOfCols, float *deviceMat);

#endif
