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

// bandwidth = read bytes / time
__device__ __forceinline__
    uint4
    load_shared_128(const uint4 *ptr)
{
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(ptr));

    uint4 value;

    asm volatile(
        "ld.volatile.shared.v4.u32 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(value.x),
          "=r"(value.y),
          "=r"(value.z),
          "=r"(value.w)
        : "r"(address)
        : "memory");

    return value;
}

__global__ void read_kernel(int repeats)
{
    extern __shared__ __align__(16) uint4 memory[];

    const int tid = threadIdx.x;

#pragma unroll 1
    for (int i = 0; i < repeats; ++i)
    {
        uint4 tmp0 = load_shared_128(&memory[tid]);
        uint4 tmp1 = load_shared_128(&memory[tid]);

        // 让两个load的结果同时保持为“被使用”，避免编译器过早复用
        // 同一组目标寄存器。空asm不会生成实际机器指令。
        asm volatile(
            ""
            :
            : "r"(tmp0.x), "r"(tmp0.y),
              "r"(tmp0.z), "r"(tmp0.w),
              "r"(tmp1.x), "r"(tmp1.y),
              "r"(tmp1.z), "r"(tmp1.w));
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
    int dyn_shared_bytes = num_thread * sizeof(uint4);
    int repeats = 1000;

    // warmup

    for (int i = 0; i < warmup; i++)
    {
        read_kernel<<<num_cta, num_thread, dyn_shared_bytes>>>(repeats);
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
        read_kernel<<<num_cta, num_thread, dyn_shared_bytes>>>(repeats);
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