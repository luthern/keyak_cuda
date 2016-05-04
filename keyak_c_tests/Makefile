src = $(wildcard *.c)
other = $(wildcard *.inc)
other += $(wildcard *.h)
obj = $(src:.c=.o)

name=keyak

CFLAGS= -O0 -g 
LDFLAGS= -lcrypto

CC=gcc

$(name): $(obj)
		$(CC) -o $@ $^ $(LDFLAGS) 

#keccak.o: keccak.cu
#	$(CCG) keccak.cu -c -o keccak.o
#
#	#piston.o: piston.cu
#	#	$(CCG) piston.cu -c -o piston.o

clean:
	rm -f $(obj) $(name)
