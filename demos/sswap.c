#include "../glblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 1048576 * 64

int main() 
{
    // create a pbuffer of size 4096x4096x4
    glblasStatus_t status;
    glblasHandle_t ctx;

    assert((status = glblasCreate(&ctx, 4096, 4096)) == GLBLAS_STATUS_SUCCESS);

    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.f;
        b[i] = 2.f;
    }

    glblasMemory_t dA = glblasMalloc(ctx, N * sizeof(float));
    glblasMemory_t dB = glblasMalloc(ctx, N * sizeof(float));
    
    glblasMemcpy(dA, a, N * sizeof(float), glblasMemcpyInfer);
    glblasMemcpy(dB, b, N * sizeof(float), glblasMemcpyInfer);
    
    glblasSswap(N, dA, 3, dB, 5);
    
    glblasMemcpy(a, dA, N * sizeof(float), glblasMemcpyInfer);
    glblasMemcpy(b, dB, N * sizeof(float), glblasMemcpyInfer);

    // automatically frees buffers, user may use `glblasFree` instead
    glblasDestroy(ctx);

    for (int i = 0; i < 16; i++)
        printf("a[%d] = %f\n", i, a[i]);
    for (int i = 0; i < 16; i++)
        printf("b[%d] = %f\n", i, b[i]);

    free(a);

    return 0;
}