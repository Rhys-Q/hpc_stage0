#include <cuda_runtime.h>
#include <iostream>
#include <vector>
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

// cta 数量固定

__global__ void read_uint4_kernel(uint4 *src, unsigned int *output, size_t count)
{
    size_t stride = gridDim.x * blockDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sum = 0;
    for (size_t i = tid; i < count; i += stride)
    {
        uint4 val = src[i];
        sum += val.x + val.y + val.z + val.w;
    }
    output[tid] = sum;
}

__global__ void read_int_kernel(int *src, unsigned int *output, size_t count)
{
    size_t stride = gridDim.x * blockDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sum = 0;
    for (size_t i = tid; i < count; i += stride)
    {
        int val = src[i];
        sum += val;
    }
    output[tid] = sum;
}

__global__ void read_uint2_kernel(uint2 *src, unsigned int *output, size_t count)
{
    size_t stride = gridDim.x * blockDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sum = 0;
    for (size_t i = tid; i < count; i += stride)
    {
        uint2 val = src[i];
        sum += val.x + val.y;
    }
    output[tid] = sum;
}

__global__ void read_char_kernel(char *src, unsigned int *output, size_t count)
{
    size_t stride = gridDim.x * blockDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sum = 0;
    for (size_t i = tid; i < count; i += stride)
    {
        char val = src[i];
        sum += val;
    }
    output[tid] = sum;
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
    int device = 0;

    CUDA_CHECK(cudaSetDevice(device));

    // get num_sm
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int num_sm = prop.multiProcessorCount;
    std::cout << "device number of sm: " << num_sm << std::endl;

    // 4 cta per sm
    int num_cta = num_sm * 4;
    int num_thread = 256;

    enum READ_MODE
    {
        CHAR,
        INT,
        UINT2,
        UINT4
    };
    READ_MODE read_mode = READ_MODE::UINT2;

    // int read_mode = 1; // 0 is char, 1 is int, 2 is uint2, 3 is uint4
    // element count
    std::vector<int> num_elements = {1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25, 1 << 26};
    for (int k = 0; k < num_elements.size(); k++)
    {
        int count = num_elements[k];
        if (read_mode == READ_MODE::INT)
        {
            count *= 4;
        }
        else if (read_mode == READ_MODE::UINT2)
        {
            count *= 2;
        }
        else if (read_mode == READ_MODE::CHAR)
        {
            count *= 4 * 4;
        }
        size_t src_bytes;
        size_t out_bytes = num_thread * sizeof(unsigned int);
        void *src = nullptr;
        if (read_mode == READ_MODE::INT)
        {
            src_bytes = count * sizeof(float);
        }
        else if (read_mode == READ_MODE::UINT2)
        {
            src_bytes = count * sizeof(uint2);
        }
        else if (read_mode == READ_MODE::CHAR)
        {
            src_bytes = count * sizeof(char);
        }
        else
        {
            src_bytes = count * sizeof(uint4);
        }

        unsigned int *out = nullptr;
        CUDA_CHECK(cudaMalloc(&src, src_bytes));
        CUDA_CHECK(cudaMalloc(&out, out_bytes));
        // warmup
        int num_warmup = 10;
        for (int i = 0; i < num_warmup; i++)
        {
            if (read_mode == READ_MODE::INT)
            {
                read_int_kernel<<<num_cta, num_thread>>>((int *)src, out, count);
            }
            else if (read_mode == READ_MODE::UINT2)
            {
                read_uint2_kernel<<<num_cta, num_thread>>>((uint2 *)src, out, count);
            }
            else if (read_mode == READ_MODE::CHAR)
            {
                read_char_kernel<<<num_cta, num_thread>>>((char *)src, out, count);
            }
            else
            {
                read_uint4_kernel<<<num_cta, num_thread>>>((uint4 *)src, out, count);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // measure

        int iters = 1000;
        std::vector<float> latency(iters, 0);
        std::vector<float> bandwidth(iters, 0);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for (int i = 0; i < iters; i++)
        {
            cudaEventRecord(start);
            if (read_mode == READ_MODE::INT)
            {
                read_int_kernel<<<num_cta, num_thread>>>((int *)src, out, count);
            }
            else if (read_mode == READ_MODE::UINT2)
            {
                read_uint2_kernel<<<num_cta, num_thread>>>((uint2 *)src, out, count);
            }
            else if (read_mode == READ_MODE::CHAR)
            {
                read_char_kernel<<<num_cta, num_thread>>>((char *)src, out, count);
            }
            else
            {
                read_uint4_kernel<<<num_cta, num_thread>>>((uint4 *)src, out, count);
            }
            cudaEventRecord(stop);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventElapsedTime(&latency[i], start, stop);
            bandwidth[i] = src_bytes * 1.0 / 1e9 / latency[i] * 1e3;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto res = summarize<float>(bandwidth);
        std::cout << "read " << std::setw(10) << src_bytes * 1.0 / 1e6 << " MB ";
        print_bandwidth(res);
    }
}