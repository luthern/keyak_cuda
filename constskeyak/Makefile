src = $(wildcard *.c)
obj = $(src:.c=.o)

name=keyak

CFLAGS= -O3 -Werror
LDFLAGS= -lcudart -lcudart

CC=g++
CCG=nvcc

$(name): $(obj) add.o
	$(CC) -o $@ $^ $(LDFLAGS) 

add.o:
	$(CCG) add.cu -c -o add.o

clean:
	rm -f $(obj) $(name)
