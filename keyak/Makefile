src = $(wildcard *.c)
obj = $(src:.c=.o)

name=keyak

$(name): $(obj)
	$(CC) -O3 -Wall -Werror -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(obj) $(name)
