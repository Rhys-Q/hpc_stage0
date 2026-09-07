#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error__ = (call);                                          \
        if (error__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(error__) << std::endl;    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


__global__ void empty_kernel(){}
 
struct Stats {
    double min=0.0;
    double mean=0.0;
    double median=0.0;
    double p99=0.0;
    double max=0.0;
};

Stats summarize(std::vector<double> values) {
    if (values.empty()) {
        return {};
    }

    std::sort(values.begin(), values.end());

    Stats stats;
    stats.min = values.front();
    stats.max = values.back();
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                 static_cast<double>(values.size());

    const auto percentile = [&values](double fraction) {
        const double index = fraction * static_cast<double>(values.size() - 1);
        const auto lower = static_cast<std::size_t>(std::floor(index));
        const auto upper = static_cast<std::size_t>(std::ceil(index));
        const double weight = index - static_cast<double>(lower);
        return values[lower] * (1.0 - weight) + values[upper] * weight;
    };

    stats.median = percentile(0.50);
    stats.p99 = percentile(0.99);
    return stats;
}


void print_stats(const std::string& name, const Stats& stats) {
    std::cout << std::left << std::setw(28) << name
              << "min=" << std::setw(10) << std::fixed << std::setprecision(3)
              << stats.min << " us  mean=" << std::setw(10) << stats.mean
              << " us  median=" << std::setw(10) << stats.median
              << " us  p99=" << std::setw(10) << stats.p99
              << " us  max=" << stats.max << " us\n";
}


int main(int argc, char** argv){
    int warmup = 10;
    int iterations = 10000;
    
    if (argc > 1){
        iterations = std::atoi(argv[1]);
    }
    if (argc > 2){
        warmup = std::atoi(argv[2]);
    }
    if (iterations <= 0 || warmup < 0) {
        std::cerr << "Usage: " << argv[0]
                  << " [iterations=10000] [warmup=100]\n";
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    std::cout << "Device: " << properties.name << "\n";
    std::cout << "Iterations: " << iterations << ", warmup: " << warmup
              << "\n\n";

    for (int i = 0; i < warmup; ++i) {
        empty_kernel<<<1, 1>>>();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> host_enqueue_us;
    host_enqueue_us.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto start = std::chrono::steady_clock::now();
        empty_kernel<<<1, 1>>>();
        const auto end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGetLastError());

        const std::chrono::duration<double, std::micro> elapsed = end - start;
        host_enqueue_us.push_back(elapsed.count());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    print_stats("Host enqueue only", summarize(host_enqueue_us));

    return EXIT_SUCCESS;
}