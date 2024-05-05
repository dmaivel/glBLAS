# glBLAS ![license](https://img.shields.io/badge/license-MIT-blue)

A software library containing BLAS functions written in OpenGL fragment shaders.

Write-up @ https://dmaivel.com/posts/outperform-cublas-with-opengl/

## Getting started

The following libraries are required for building `glBLAS`:
- libepoxy
- EGL

```bash
git clone https://github.com/dmaivel/glBLAS.git
cd glBLAS
make
```

### Usage

For use in projects, simply include `glblas.c` and `glblas.h`, and link with `epoxy` & `m`.

## Kernels

- Level 1
  - sswap
  - sscal
  - scopy
  - saxpy
  - sdot
  - sasum
- Level 3
  - sgemm