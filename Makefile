# Makefile for cuSAE

# Change to appropriate paths
MATLAB_ROOT=/usr/local/MATLAB/R2012a
CUDA_LIB=/usr/local/cuda/lib64/

# mex compiler configuration
MEX=mex -O -cxx
MEX_INC=$(MATLAB_ROOT)/extern/include/

# nvcc compiler configuration
NVCC = nvcc -O4
LIBS = -lcublas -lcuda -lcudart -largeArrayDims

all: computation_functions.o sparseAutoencoderCost.o 

	$(MEX) sparseAutoencoderCost.o computation_functions.o -L${CUDA_LIB} $(LIBS) -o sparseAutoencoderCost 

computation_functions.o: computation_functions.cu computation_functions.h

	$(NVCC) -c -Xcompiler -fPIC computation_functions.cu -I$(MEX_INC)

sparseAutoencoderCost.o: sparseAutoencoderCost.cu

	$(NVCC) -c -Xcompiler -fPIC sparseAutoencoderCost.cu -I$(MEX_INC)

clean:
	rm -f *.o *.out *.exe
	rm -f *.bin
	rm -f *~
