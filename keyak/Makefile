src = $(wildcard *.cpp)
obj = $(src:.cpp=.o) engine.o utils.o

name=keyak

CPPFLAGS= -I/usr/local/cuda-7.5/include/ -I $(CUDA_INC) -O3 -DBUF_SLOTS=$(BUF_SLOTS)
CGFLAGS= -I/usr/local/cuda-7.5/include/ -I $(CUDA_INC) -DBUF_SLOTS=$(BUF_SLOTS) -O3
LDFLAGS= -lcudart -lcudart -lcrypto

CC=nvcc
CXX=nvcc
CCG=nvcc

$(name): $(obj)
	$(CCG) -o $@ $^ $(LDFLAGS) 

engine.o: engine.cu piston.cu keccak.cu engine.h defs.h piston.h
	$(CCG) engine.cu -c $(CGFLAGS) -o engine.o

utils.o: utils.cu defs.h
	$(CCG) utils.cu -c $(CGFLAGS) -o utils.o

#keccak.o: keccak.cu
#	$(CCG) keccak.cu -c -o keccak.o

#piston.o: piston.cu
#	$(CCG) piston.cu -c -o piston.o

clean:
	rm -f $(obj) $(name)
