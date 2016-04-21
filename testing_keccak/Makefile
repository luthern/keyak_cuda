
test: main.o keccak_ref.o keccak_cuda.o misc.o
	nvcc -o $@ $^

main.o: main_basic.cu
	nvcc -o $@ -c $^ $(CFLAGS)

keccak_cuda.o: cuda_keccak_basic.cu
	nvcc -o $@ -c $^ $(CFLAGS)

keccak_ref.o: keccak.c
	g++ -o $@ -c $^ $(CFLAGS)

misc.o: misc.c
	g++ -o $@ -c $^ $(CFLAGS)

clean:
	rm -f *.o test
