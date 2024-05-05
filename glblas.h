// Copyright (c) 2023 dmaivel

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stddef.h>

typedef enum glblasMemcpyKind {
    glblasMemcpyInfer,
    glblasMemcpyDeviceToHost,
    glblasMemcpyHostToDevice,
    glblasMemcpyDeviceToDevice
} glblasMemcpyKind_t;

typedef enum glblasOperation {
    GLBLAS_OP_N,
    GLBLAS_OP_T
} glblasOperation_t;

typedef enum glblasStatus {
    GLBLAS_STATUS_SUCCESS,
    GLBLAS_STATUS_ALLOC_FAILED,
    GLBLAS_STATUS_INVALID_VALUE,
    GLBLAS_STATUS_NOT_SUPPORTED,
    GLBLAS_STATUS_EXECUTION_FAILED,
    GLBLAS_STATUS_DIMENSION_OVERFLOW,
} glblasStatus_t;

typedef void *glblasHandle_t;
typedef void *glblasMemory_t;

#ifdef __cplusplus
extern "C" {
#endif

glblasStatus_t glblasCreate(glblasHandle_t *handle, int width, int height);
void glblasSync();
void glblasDestroy(glblasHandle_t ctx);

glblasMemory_t glblasMalloc(glblasHandle_t ctx, size_t size);
glblasStatus_t glblasMemcpy(void *dst, void *src, size_t size, glblasMemcpyKind_t kind);
void glblasFree(glblasMemory_t buf);

// swap x & y
glblasStatus_t glblasSswap(int N, glblasMemory_t x, int incx, glblasMemory_t y, int incy);

// x = a*x
glblasStatus_t glblasSscal(int N, const float alpha, glblasMemory_t x, int incx);

// copy x into y
glblasStatus_t glblasScopy(int N, const glblasMemory_t x, int incx, glblasMemory_t y, int incy);

// y = a*x + y
glblasStatus_t glblasSaxpy(int N, const float alpha, const glblasMemory_t x, int incx, glblasMemory_t y, int incy);

// dot product
glblasStatus_t glblasSdot(int N, glblasMemory_t result, const glblasMemory_t x, int incx, const glblasMemory_t y, int incy);

// sum of abs values
glblasStatus_t glblasSasum(int N, glblasMemory_t result, const glblasMemory_t x, int incx);

// matrix matrix multiply
glblasStatus_t glblasSgemm( glblasOperation_t transa, glblasOperation_t transb
                          , int M, int N, int K, const float alpha
                          , const glblasMemory_t a, const int lda
                          , const glblasMemory_t b, const int ldb, const float beta
                          , glblasMemory_t c, const int ldc );

// optimized matrix multiply
glblasStatus_t glblasSgemm4x4( glblasOperation_t transa, glblasOperation_t transb
                             , int M, int N, int K, const float alpha
                             , const glblasMemory_t a, const int lda
                             , const glblasMemory_t b, const int ldb, const float beta
                             , glblasMemory_t c, const int ldc );

#ifdef __cplusplus
}
#endif