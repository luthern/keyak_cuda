# keyak_cuda
Implementation of Keyak Cipher in C/CUDA

Keccak Implementation based on the prior work of Gerhard Hoffmann.

## Prerequisites

* Cuda/NVidia enabled machine.  

Should work on any NVidia architecture.  Tested at home and on Hokiespeed.

## Compiling Keyak

```bash
cd keyak/
export BUF_SLOTS=74
# only if CUDA_INC is not defined
export CUDA_INC=.
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

## Usage

```bash
./keyak
usage: ./keyak <key-hex> <input-file> <output-file> [-n <nonce-hex>] [-m <metadata-hex>] [-i <iterations>]
```
