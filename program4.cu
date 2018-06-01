#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <random>
#include <cstdint>

void fill_matrix(float* matrix, uint64_t n);
void print_matrix(float* matrix, uint64_t n);

// Uses 1 block with a 1D dimension; assumes total number of threads = N
// Has 2xNxNxN total floating-point operations
__global__ void gpu_basic_mm(float* matrix1, float* matrix2, float* result, uint64_t n)
{
    // Divide threads into indices (assuming 1D blocks)
    //uint64_t num_threads = gridDim.x * blockDim.x;
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < n*n; ++i)
    {
        float val = matrix1[i];
    }

    printf("hey\n");

    // Assume num threads = n
    for (int row = 0; row < n; ++row)
    {
        float sum = 0;
        for (int item = 0; item < n; ++item)
        {
            sum += matrix1[row * n + item] * matrix2[item * n + thread_index];
        }
        result[thread_index * n + row] = sum;
    }
}

int main()
{
    // Size of matrices
    uint64_t n = 1024;

    float* m1 = new float[n * n];
    float* m2 = new float[n * n];
    float* result = new float[n * n];

    float* g_m1;
    float* g_m2;
    float* g_result;
    cudaMalloc((void**)&g_m1, n * n * sizeof(float));
    cudaMalloc((void**)&g_m2, n * n * sizeof(float));
    cudaMalloc((void**)&g_result, n * n * sizeof(float));

    fill_matrix(m1, n);
    fill_matrix(m2, n);

    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        printf("Allocation Error: %s\n", cudaGetErrorString(code));
    }

    //printf("-----m1-----\n");
    //print_matrix(m1, n);

    //printf("-----m2-----\n");
    //print_matrix(m2, n);


    // Timer start including memcpy operations
    //auto start = std::chrono::system_clock::now();

    cudaMemcpy(g_m1, m1, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_m2, m2, n * n * sizeof(float), cudaMemcpyHostToDevice);

    code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        printf("Memcpy Error: %s\n", cudaGetErrorString(code));
    }

    // Timer start excluding memcpy operations
    clock_t start = clock();

    // has 2xnxnxn total floating-point operations
    gpu_basic_mm<<<4, 1024>>>(g_m1, g_m2, g_result, n);
    cudaThreadSynchronize();

    // Timer end excluding memcpy operations
    float elapsed_seconds = (float)(clock() - start) / CLOCKS_PER_SEC;

    code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        printf("Kernel Error: %s\n", cudaGetErrorString(code));
    }

    cudaMemcpy(result, g_result, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Timer end including memcpy operations
    //float elapsed_seconds = (std::chrono::system_clock::now() - start).count();


    //printf("-----result-----\n");
    //print_matrix(result, n);

    code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        printf("Memcpy Error 2: %s\n", cudaGetErrorString(code));
    }

    float flops = (2 * n * n * n) / elapsed_seconds;
    printf("Operations: %lld\n", (2 * n * n * n));
    printf("Seconds: %f\n", elapsed_seconds);
    printf("FLOPS for gpu_basic_mm() at size %lld matrices = %f", n, flops);


    cudaFree(m1);
    cudaFree(m2);
    cudaFree(g_result);

    delete[] m1;
    delete[] m2;
    delete[] result;

}

void fill_matrix(float* matrix, uint64_t n)
{
    std::uniform_real_distribution<float> distribution(2, 100);
    std::default_random_engine generator;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix[i * n + j] = distribution(generator);
        }
    }
}

void print_matrix(float* matrix, uint64_t n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f\t", matrix[i * n + j]);
        }
        printf("\n");
    }
}
