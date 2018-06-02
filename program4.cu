#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <random>
#include <cstdint>
#include <iostream>
#include <cstring>

#define MATRIX_SIZE 4096
#define BLOCK_SIZE 32

void fill_matrix(float* matrix, uint64_t n);
void print_matrix(float* matrix, uint64_t n);

// Uses 1 block with a 1D dimension; assumes total number of threads = N
// Has 2xNxNxN total floating-point operations
__global__ void gpu_basic_mm(float* matrix1, float* matrix2, float* result, uint64_t n)
{
    // Divide threads into indices (assuming 1D blocks)
    //int num_threads = gridDim.x * blockDim.x;
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

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

// Assumes blockDim = NxA and threadDim = B where A * B = N
// Has 2xNxNxN total floating-point operations
// Uses N*N total threads to calculate result
__global__ void gpu_better_mm(float* matrix1, float* matrix2, float* result, uint64_t n)
{
    int row = blockIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.x;

    // Assume num threads = n * n
    float sum = 0;
    for (int item = 0; item < n; ++item)
    {
        sum += matrix1[row * n + item] * matrix2[item * n + column];
    }
    result[column * n + row] = sum;
}

// Uses 1 block with a 1D dimension; assumes total number of threads = N
// Has 2xNxNxN total floating-point operations
// A simple improvement on gpu_basic_mm() using shared memory
__global__ void gpu_shared_mm(float* matrix1, float* matrix2, float* result, uint64_t n)
{
    // Divide threads into indices (assuming 1D blocks)
    //int num_threads = gridDim.x * blockDim.x;
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shared_row[MATRIX_SIZE];
    __shared__ float shared_column[MATRIX_SIZE];

    // Assume num threads = n
    for (int row = 0; row < n; ++row)
    {
        shared_row[thread_index] = matrix1[row * n + thread_index];
        shared_column[thread_index] = matrix2[thread_index * n + thread_index];
        float sum = 0;
        for (int item = 0; item < n; ++item)
        {
            sum += shared_row[thread_index] * shared_column[thread_index];
        }
        result[thread_index * n + row] = sum;
    }
}

// Assumes 2D blocks and 2D threads, where
// BlockDim.x * ThreadDim.x = N
// BlockDim.y * ThreadDim.y = N
__global__ void gpu_optimized_mm(float* matrix1, float* matrix2, float* result, int n)
{
    // Assumes square tiles
    int tile_size = blockDim.x;

    // Block row and column
    int tile_row = blockIdx.y;
    int tile_column = blockIdx.x;

    // this block's tile
    float* tile = &result[tile_size * tile_size * tile_row + tile_size * tile_column];

    float value = 0;

    // Thread row and column for this tile
    int row = threadIdx.y;
    int col = threadIdx.x;


    for (int i = 0; i < (n / tile_size); ++i)
    {
        float *temp1 = &matrix1[tile_size * tile_size * tile_row + tile_size * i];
        float *temp2 = &matrix2[tile_size * tile_size * i + tile_size * tile_column];

        __shared__ float shared1[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shared2[BLOCK_SIZE][BLOCK_SIZE];

        shared1[row][col] = temp1[row * tile_size + col];
        shared2[row][col] = temp2[row * tile_size + col];

        __syncthreads();

        for (int j = 0; j < tile_size; ++j)
        {
            value += shared1[row][j] * shared2[j][col];
        }

        __syncthreads();
    }

    tile[row * tile_size + col] = value;
}

int main()
{
    // Size of matrices
    uint64_t n = MATRIX_SIZE;

    float* m1 = new float[n * n];
    float* m2 = new float[n * n];
    float* result = new float[n * n];

    float* g_m1;
    float* g_m2;
    float* g_result;
    cudaMalloc(reinterpret_cast<void**>(&g_m1), n * n * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&g_m2), n * n * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&g_result), n * n * sizeof(float));

    fill_matrix(m1, n);
    fill_matrix(m2, n);

    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        printf("Allocation Error: %s\n", cudaGetErrorString(code));
    }

    // Timer start including memcpy operations
    //clock_t start = clock();

    cudaMemcpy(g_m1, m1, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_m2, m2, n * n * sizeof(float), cudaMemcpyHostToDevice);

    code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        printf("Memcpy Error: %s\n", cudaGetErrorString(code));
    }

    // Timer start excluding memcpy operations
    clock_t start = clock();

    //gpu_basic_mm<<<32, 32>>>(g_m1, g_m2, g_result, n);
    //gpu_better_mm<<<dim3(256, 4), 1024, 2 * n * n>>>(g_m1, g_m2, g_result, n);
    gpu_optimized_mm<<<dim3(128, 128), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(g_m1, g_m2, g_result, n);

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
    //float elapsed_seconds = (float)(clock() - start) / CLOCKS_PER_SEC;

    code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        printf("Memcpy Error 2: %s\n", cudaGetErrorString(code));
    }

    float flops = (2 * n * n * n) / elapsed_seconds;
    std::cout << "Operations: " << (2 * n * n * n) << std::endl;
    printf("Seconds: %f\n", elapsed_seconds);
    std::cout << "FLOPS for gpu_basic_mm() at size " << n << " matrices = " << flops << std::endl;

    cudaFree(m1);
    cudaFree(m2);
    cudaFree(g_result);

    delete[] m1;
    delete[] m2;
    delete[] result;
}

// Fill matrix with random floats from 2 - 100
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

// Print matrix (for debugging purposes)
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
