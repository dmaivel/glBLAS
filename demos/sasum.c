#include "../glblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 16

int main() 
{
    // create a pbuffer of size 512x512x4
    glblasStatus_t status;
    glblasHandle_t ctx;

    assert((status = glblasCreate(&ctx, 512, 512)) == GLBLAS_STATUS_SUCCESS);

    float *a = malloc(N * sizeof(float));
    float *b = malloc(1 * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
    }

    glblasMemory_t dA = glblasMalloc(ctx, N * sizeof(float));
    glblasMemory_t dB = glblasMalloc(ctx, 1 * sizeof(float));
    
    glblasMemcpy(dA, a, N * sizeof(float), glblasMemcpyInfer);
    
    glblasSasum(N, dB, dA, 1);
    
    glblasMemcpy(b, dB, 1 * sizeof(float), glblasMemcpyInfer);

    // automatically frees buffers, user may use `glblasFree` instead
    glblasDestroy(ctx);

    printf("b = %f\n", b[0]);

    free(a);

    return 0;
}