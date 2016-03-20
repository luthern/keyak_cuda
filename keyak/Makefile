src = $(wildcard *.c)
obj = $(src:.c=.o) add.o engine.o

name=keyak

CFLAGS= -O3 -Werror
LDFLAGS= -lcudart -lcudart

CC=g++
CCG=nvcc

$(name): $(obj)
	$(CC) -o $@ $^ $(LDFLAGS) 

add.o:
	$(CCG) add.cu -c -o add.o

engine.o: engine.cu piston.cu
	$(CCG) engine.cu -c -o engine.o

#piston.o: piston.cu
#	$(CCG) piston.cu -c -o piston.o

clean:
	rm -f $(obj) $(name)
