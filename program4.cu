#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <chrono>
#include <random>

void fill_matrix(double* matrix, int n);
void print_matrix(double* matrix, int n);

// Uses 1 block with a 1D dimension; assumes total number of threads = N
// Has 2xNxNxN total floating-point operations
__global__ void gpu_basic_mm(double* matrix1, double* matrix2, double* result, int n)
{
    // Divide threads into indices (assuming 1D blocks)
    //int num_threads = gridDim.x * blockDim.x;
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

    // Assume num threads = n
    for (int row = 0; row < n; ++row)
    {
        double sum = 0;
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
    int n = 1024;

    double* m1 = new double[n * n];
    double* m2 = new double[n * n];
    double* result = new double[n * n];

    double* g_m1;
    double* g_m2;
    double* g_result;
    cudaMalloc((void**)&g_m1, n * n * sizeof(double));
    cudaMalloc((void**)&g_m2, n * n * sizeof(double));
    cudaMalloc((void**)&g_result, n * n * sizeof(double));

    fill_matrix(m1, n);
    fill_matrix(m2, n);

    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        printf("asd\n");
        printf("An error occurred: %s\n", cudaGetErrorString(code));
    }

    //printf("-----m1-----\n");
    //print_matrix(m1, n);

    //printf("-----m2-----\n");
    //print_matrix(m2, n);


    // Timer start including memcpy operations
    //auto start = std::chrono::system_clock::now();

    cudaMemcpy(g_m1, m1, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_m2, m2, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Timer start excluding memcpy operations
    clock_t start = clock();

    // has 2xnxnxn total floating-point operations
    gpu_basic_mm<<<32, 32>>>(g_m1, g_m2, g_result, n);
    cudaThreadSynchronize();

    // Timer end excluding memcpy operations
    double elapsed_seconds = (double)(clock() - start) / CLOCKS_PER_SEC;

    cudaMemcpy(result, g_result, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Timer end including memcpy operations
    //double elapsed_seconds = (std::chrono::system_clock::now() - start).count();


    //printf("-----result-----\n");
    //print_matrix(result, n);

    code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        printf("An error occurred: %s\n", cudaGetErrorString(code));
    }

    long operations = n;
    operations = 2 * operations * operations * operations;
    double flops = operations / elapsed_seconds;
    printf("Operations: %d\n", operations);
    printf("Seconds: %f\n", elapsed_seconds);
    printf("FLOPS for gpu_basic_mm() at size %d matrices = %f", n, flops);


    cudaFree(m1);
    cudaFree(m2);
    cudaFree(g_result);

    delete[] m1;
    delete[] m2;
    delete[] result;

}

void fill_matrix(double* matrix, int n)
{
    std::uniform_real_distribution<double> distribution(2, 100);
    std::default_random_engine generator;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix[i * n + j] = distribution(generator);
        }
    }
}

void print_matrix(double* matrix, int n)
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
