src = $(wildcard *.c)
obj = $(src:.c=.o) engine.o utils.o

name=keyak

CFLAGS= -O3 -Werror
LDFLAGS= -lcudart -lcudart

CC=g++
CCG=nvcc

$(name): $(obj)
	$(CCG) -o $@ $^ $(LDFLAGS) 

engine.o: engine.cu piston.cu keccak.cu
	$(CCG) engine.cu -c -o engine.o

utils.o: utils.cu
	$(CCG) utils.cu -c -o utils.o

#keccak.o: keccak.cu
#	$(CCG) keccak.cu -c -o keccak.o

#piston.o: piston.cu
#	$(CCG) piston.cu -c -o piston.o

clean:
	rm -f $(obj) $(name)
