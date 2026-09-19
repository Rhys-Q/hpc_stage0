#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(call)                                                                                                   \
    do                                                                                                                     \
    {                                                                                                                      \
        cudaError_t error = (call);                                                                                        \
        if (error != cudaSuccess)                                                                                          \
        {                                                                                                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE);                                                                                       \
        }                                                                                                                  \
    } while (0);

__global__ void write_kernel_double(int repeats)
{
    extern __shared__ volatile double memory1[];

    const int tid = threadIdx.x;
    double tmp = tid;

#pragma unroll 1
    for (int i = 0; i < repeats; ++i)
    {
        memory1[tid] = tmp;
        memory1[tid] = tmp;
    }
}

template <typename DataType>
std::vector<DataType> summarize(std::vector<DataType> values)
{
    if (values.size() == 0)
    {
        return {};
    }
    DataType minv = values[0];
    DataType maxv = values[0];
    DataType sumv = values[0];

    for (int i = 1; i < values.size(); i++)
    {
        minv = min(values[i], minv);
        maxv = max(values[i], maxv);
        sumv += values[i];
    }
    DataType mean = sumv / values.size();
    return {minv, maxv, mean};
}

void print_bandwidth(std::vector<float> bandwidth)
{
    std::cout << "mean :" << std::setw(10) << bandwidth[2] << " min :" << std::setw(10) << bandwidth[0] << " max: " << std::setw(10) << bandwidth[1] << std::endl;
}

int main()
{

    int warmup = 100;
    int iters = 1000;

    // std::vector<int> counts;

    // get num_sm
    cudaDeviceProp prop;
    int device = 0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int num_sm = prop.multiProcessorCount;
    std::cout << "device number of sm: " << num_sm << std::endl;

    // 4 cta per sm
    int num_cta = num_sm * 8;
    int num_thread = 1024;
    int dyn_shared_bytes = num_thread * sizeof(double);
    int repeats = 1000;

    // warmup

    for (int i = 0; i < warmup; i++)
    {
        write_kernel_double<<<num_cta, num_thread, dyn_shared_bytes>>>(repeats);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // measure
    std::vector<float> bandwidth;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++)
    {
        cudaEventRecord(start);
        write_kernel_double<<<num_cta, num_thread, dyn_shared_bytes>>>(repeats);
        cudaEventRecord(stop);
        CUDA_CHECK(cudaDeviceSynchronize());
        float time_used = 0;
        cudaEventElapsedTime(&time_used, start, stop);
        float bw = (double)(repeats) * 2 * num_cta * dyn_shared_bytes / 1e9 / time_used * 1e3;
        bandwidth.push_back(bw);
    }

    auto res = summarize<float>(bandwidth);
    print_bandwidth(res);
}