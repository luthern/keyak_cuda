src = $(wildcard *.c)
obj = $(src:.c=.o)

name=keyak

CFLAGS= -O3 -Werror

$(name): $(obj)
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(obj) $(name)
