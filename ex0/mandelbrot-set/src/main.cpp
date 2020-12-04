#include "aux.h"
#include "data_types.h"
#include "kernels.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

void convertDataType(const std::vector<uint32_t> &interationFiled,
                     std::vector<unsigned char> &colorField, const ProgramSettins &settings) {

    auto mapIterationToColor = [&settings](size_t iteration) -> unsigned char {
        constexpr float WHITE_COLOR = 255.0;
        constexpr float BLACK_COLOR = 0.0;

        return (iteration >= settings.maxIterations)
                   ? BLACK_COLOR
                   : WHITE_COLOR - (iteration / (real)settings.maxIterations) * WHITE_COLOR;
    };

    std::transform(interationFiled.begin(), interationFiled.end(), colorField.begin(),
                   mapIterationToColor);
}

double computeGFLOPs(const std::vector<uint32_t> &interationFiled, double seconds) {
    long long totalNumIterations =
        std::accumulate(interationFiled.begin(), interationFiled.end(), static_cast<long long>(0));

    // NOTE: manually computed value
    constexpr int FLOP_PER_ITERATION = 14;
    return FLOP_PER_ITERATION * totalNumIterations / (1e9 * seconds);
}

int main(int argc, char *argv[]) {

    // get input
    ProgramSettins settings;
    try {
        settings = getInputFromCmd(argc, argv);

        constexpr unsigned HEADER_LENGTH = 80;
        std::cout << std::string(HEADER_LENGTH, '=') << std::endl;
        std::cout << settings;
        std::cout << std::string(HEADER_LENGTH, '=') << std::endl;
        if (settings.mode == MODE::GPU) {
            gpu::printHardwareInfo();
            std::cout << std::string(HEADER_LENGTH, '=') << std::endl;
        }

        checkInputData(settings);
    } catch (std::runtime_error &error) {
        std::cout << error.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    const size_t FIELD_SIZE = settings.numPixelsX * settings.numPixelsY;
    std::vector<uint32_t> field(FIELD_SIZE, real{0.0});

    // compute Mandelbrot Set
    double gpuComputeTime{0.0};
    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    switch (settings.mode) {
        case MODE::Serial: {
            serial::computeMandelbrotSet(field, settings);
            break;
        }
        case MODE::Threading: {
            threading::computeMandelbrotSet(field, settings);
            break;
        }
        case MODE::GPU: {
            gpu::computeMandelbrotSet(field, settings, gpuComputeTime);
            std::cout << "GPU compute time: " << gpuComputeTime << " seconds.\n";
            break;
        }
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
    std::cout << "Spent time: " << elapsedTime.count() << " seconds.\n";

    std::cout << "GFLOP/s (elapsed time): " << computeGFLOPs(field, elapsedTime.count())
              << std::endl;
    if (settings.mode == MODE::GPU) {
        std::cout << "GFLOP/s (compute time): " << computeGFLOPs(field, gpuComputeTime)
                  << std::endl;
    }

    // write output
    std::vector<unsigned char> outputField(FIELD_SIZE, real{0.0});
    convertDataType(field, outputField, settings);
    writeData(outputField, settings);

    return 0;
}