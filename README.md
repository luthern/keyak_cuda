# keyak_cuda
Implementation of Keyak Cipher in C/CUDA

Keccak Implementation based on the prior work of Gerhard Hoffmann.

## Prerequisites

* Cuda/NVidia enabled machine.  

Should work on any NVidia architecture.  Tested at home and on Hokiespeed.

## Compiling Keyak

```bash
cd keyak/
make -j5
```

## Testing

Run the testbench

```bash
./testbench.sh
```

Run test vector to ensure implementation is correct

```bash
cd vectors/
./v1.sh
```
