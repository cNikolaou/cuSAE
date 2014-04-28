# Makefile of cuSAE

NVCC = nvcc -O4
LIBS = -lcublas

all: sparseAutoencoderCost.o

	${NVCC} $(LIBS) sparseAutoencoderCost.obj -o cuSAE 

sparseAutoencoderCost.o: sparseAutoencoderCost.cu

	$(NVCC) $(LIBS) -c sparseAutoencoderCost.cu

clean:
	rm -f *~ *.o
