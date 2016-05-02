src = $(wildcard *.c)
obj = $(src:.c=.o) engine.o utils.o

name=keyak

CFLAGS= -I/usr/local/cuda-7.5/include/ -I $(CUDA_INC) -O3 -Werror 
LDFLAGS= -lcudart -lcudart -lcrypto

CC=g++
CCG=nvcc

$(name): $(obj)
	$(CCG) -o $@ $^ $(LDFLAGS) 

engine.o: engine.cu piston.cu keccak.cu engine.h defs.h piston.h
	$(CCG) engine.cu -c -o engine.o

utils.o: utils.cu defs.h
	$(CCG) utils.cu -c -o utils.o

#keccak.o: keccak.cu
#	$(CCG) keccak.cu -c -o keccak.o

#piston.o: piston.cu
#	$(CCG) piston.cu -c -o piston.o

clean:
	rm -f $(obj) $(name)
