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

    for (int i = 0; i < N; i++) {
        a[i] = 1.f;
    }

    glblasMemory_t dA = glblasMalloc(ctx, N * sizeof(float));
    
    glblasMemcpy(dA, a, N * sizeof(float), glblasMemcpyInfer);
    
    glblasSscal(N, 2, dA, 4);
    
    glblasMemcpy(a, dA, N * sizeof(float), glblasMemcpyInfer);

    // automatically frees buffers, user may use `glblasFree` instead
    glblasDestroy(ctx);

    for (int i = 0; i < 16; i++)
        printf("a[%d] = %f\n", i, a[i]);

    free(a);

    return 0;
}