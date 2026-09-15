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

__global__ void write_uint4_kernel(uint4 *src, size_t count)
{
    size_t stride = gridDim.x * blockDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < count; i += stride)
    {
        uint4 tmp;
        tmp.x = tid;
        tmp.y = tid;
        tmp.z = tid;
        tmp.w = tid;
        src[i] = tmp;
    }
}

__global__ void write_int_kernel(int *src, size_t count)
{
    size_t stride = gridDim.x * blockDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < count; i += stride)
    {
        src[i] = tid;
    }
}

__global__ void write_uint2_kernel(uint2 *src, size_t count)
{
    size_t stride = gridDim.x * blockDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < count; i += stride)
    {
        uint2 tmp;
        tmp.x = tid;
        tmp.y = tid;
        src[i] = tmp;
    }
}
__global__ void write_char_kernel(char *src, size_t count)
{
    size_t stride = gridDim.x * blockDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < count; i += stride)
    {
        src[i] = tid;
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

    enum WRITE_MODE
    {
        CHAR,
        INT,
        UINT2,
        UINT4
    };
    WRITE_MODE write_mode = WRITE_MODE::UINT2;
    // element count
    std::vector<int> num_elements = {1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25, 1 << 26};
    for (int k = 0; k < num_elements.size(); k++)
    {
        int count = num_elements[k];
        if (write_mode == WRITE_MODE::INT)
        {
            count *= 4;
        }
        else if (write_mode == WRITE_MODE::UINT2)
        {
            count *= 2;
        }
        else if (write_mode == WRITE_MODE::CHAR)
        {
            count *= 4 * 4;
        }
        size_t src_bytes;
        void *src = nullptr;
        if (write_mode == WRITE_MODE::INT)
        {
            src_bytes = count * sizeof(float);
        }
        else if (write_mode == WRITE_MODE::UINT2)
        {
            src_bytes = count * sizeof(uint2);
        }
        else if (write_mode == WRITE_MODE::CHAR)
        {
            src_bytes = count * sizeof(char);
        }
        else
        {
            src_bytes = count * sizeof(uint4);
        }

        CUDA_CHECK(cudaMalloc(&src, src_bytes));

        // warmup
        int num_warmup = 10;
        for (int i = 0; i < num_warmup; i++)
        {
            if (write_mode == WRITE_MODE::INT)
            {
                write_int_kernel<<<num_cta, num_thread>>>((int *)src, count);
            }
            else if (write_mode == WRITE_MODE::UINT2)
            {
                write_uint2_kernel<<<num_cta, num_thread>>>((uint2 *)src, count);
            }
            else if (write_mode == WRITE_MODE::CHAR)
            {
                write_char_kernel<<<num_cta, num_thread>>>((char *)src, count);
            }
            else
            {
                write_uint4_kernel<<<num_cta, num_thread>>>((uint4 *)src, count);
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
            if (write_mode == WRITE_MODE::INT)
            {
                write_int_kernel<<<num_cta, num_thread>>>((int *)src, count);
            }
            else if (write_mode == WRITE_MODE::UINT2)
            {
                write_uint2_kernel<<<num_cta, num_thread>>>((uint2 *)src, count);
            }
            else if (write_mode == WRITE_MODE::CHAR)
            {
                write_char_kernel<<<num_cta, num_thread>>>((char *)src, count);
            }
            else
            {
                write_uint4_kernel<<<num_cta, num_thread>>>((uint4 *)src, count);
            }
            cudaEventRecord(stop);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventElapsedTime(&latency[i], start, stop);
            bandwidth[i] = src_bytes * 1.0 / 1e9 / latency[i] * 1e3;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto res = summarize<float>(bandwidth);
        std::cout << "write " << std::setw(10) << src_bytes * 1.0 / 1e6 << " MB ";
        print_bandwidth(res);
    }
}