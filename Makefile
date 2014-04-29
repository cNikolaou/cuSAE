# Makefile of cuSAE

NVCC = nvcc -O4
LIBS = -lcublas

all: computation_functions.o sparseAutoencoderCost.o 

	${NVCC} $(LIBS) sparseAutoencoderCost.obj computation_functions.obj -o cuSAE 

computation_functions.o: computation_functions.cu computation_functions.h

	$(NVCC) $(LIBS) -c computation_functions.cu

sparseAutoencoderCost.o: sparseAutoencoderCost.cu

	$(NVCC) $(LIBS) -c sparseAutoencoderCost.cu

clean:
	rm -f *.o *.out *.exe
	rm -f *.bin
	rm -f *~
