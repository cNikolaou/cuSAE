#ifndef CUSAE_SRC_COMPUTATION_FUNCTIONS_H
#define CUSAE_SRC_COMPUTATION_FUNCTIONS_H

__global__ void Sigmoid(const float *a, const int numberOfElements, 
						float *sa);
__global__ void ComputeAbsDiff(const float *hx, const float *y, int N, 
							   float *diff);
__global__ void CompDelta3(const float *y, const float *a3, int length, 
						   float *delta3);
__global__ void CompDelta(const float *a2, int N, float *delta);
__global__ void SetOnes(int length, float *ones);
__global__ void SetZeros(int length, float *zeros);

void ComputeSigmoid(const float *z, int N, float *a);
void SetRepMat(const float *b, int numberOfRows, int numberOfCols, float *z);
void ComputePartCost(cublasHandle_t handle, const float *hx, const float *y, 
					 int numberOfRows, int numberOfCols, float *partCost);
void CompDelta(cublasHandle_t handle, const float *W2, const float *a2, 
			   int hiddenSize, int numberOfExamples, int visibleSize, 
			   float *delta3, float *delta2);
void CompWgrad(float *W, int numberOfRows, int numberOfCols, int m);
void Compbgrad(float *b, int numberOfRows, int m);
void PrintReturnedMat(int numberOfRows, int numberOfCols, 
					  const float *deviceMat);

#endif
