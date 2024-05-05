#include "../glblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 32

int main() 
{
    // create a pbuffer of size 16x16x4
    glblasStatus_t status;
    glblasHandle_t ctx;

    assert((status = glblasCreate(&ctx, 16, 16)) == GLBLAS_STATUS_SUCCESS);

    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(1 * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.f;
        b[i] = 2.f;
    }

    glblasMemory_t dA = glblasMalloc(ctx, N * sizeof(float));
    glblasMemory_t dB = glblasMalloc(ctx, N * sizeof(float));
    glblasMemory_t dC = glblasMalloc(ctx, 1 * sizeof(float));
    
    glblasMemcpy(dA, a, N * sizeof(float), glblasMemcpyInfer);
    glblasMemcpy(dB, b, N * sizeof(float), glblasMemcpyInfer);
    
    glblasSdot(N, dC, dA, 1, dB, 1);
    
    glblasMemcpy(c, dC, 1 * sizeof(float), glblasMemcpyInfer);

    // automatically frees buffers, user may use `glblasFree` instead
    glblasDestroy(ctx);

    printf("c = %f\n", c[0]);

    free(a);

    return 0;
}