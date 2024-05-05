#include "../glblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define M 4
#define N 4
#define K 4

int main() 
{
    // create a pbuffer of size 128x128x4
    glblasStatus_t status;
    glblasHandle_t ctx;

    assert((status = glblasCreate(&ctx, 128, 128)) == GLBLAS_STATUS_SUCCESS);

    float *a = malloc(M * K * sizeof(float));
    float *b = malloc(K * N * sizeof(float));
    float *c = malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++)
        a[i] = i + 1.f;

    for (int i = 0; i < N * K; i++)
        b[i] = i + 1.f;

    printf("a =\n");
    for (int x = 0; x < M; x++) {
        for (int y = 0; y < K; y++)
            printf("%f ", a[y * M + x]);
        puts("");
    }

    printf("b =\n");
    for (int x = 0; x < K; x++) {
        for (int y = 0; y < N; y++)
            printf("%f ", b[y * K + x]);
        puts("");
    }

    glblasMemory_t dA = glblasMalloc(ctx, M * K * sizeof(float));
    glblasMemory_t dB = glblasMalloc(ctx, K * N * sizeof(float));
    glblasMemory_t dC = glblasMalloc(ctx, M * N * sizeof(float));
    
    glblasMemcpy(dA, a, M * K * sizeof(float), glblasMemcpyInfer);
    glblasMemcpy(dB, b, K * N * sizeof(float), glblasMemcpyInfer);
    
    glblasSgemm(GLBLAS_OP_N, GLBLAS_OP_N, M, N, K, 1, dA, M, dB, K, 1, dC, M);
    
    glblasMemcpy(c, dC, M * N * sizeof(float), glblasMemcpyInfer);

    // automatically frees buffers, user may use `glblasFree` instead
    glblasDestroy(ctx);

    printf("c =\n");
    for (int x = 0; x < K; x++) {
        for (int y = 0; y < N; y++)
            printf("%f ", c[y * K + x]);
        puts("");
    }

    free(a);
    free(b);
    free(c);

    return 0;
}