#Makefile of cuSAE

MATLAB_ROOT=/usr/pkg/matlab-2013b/
CUDA_LIB=/usr/local/cuda/lib64/

MEX=mex -O
MEX_INC=$(MATLAB_ROOT)/extern/include/
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
