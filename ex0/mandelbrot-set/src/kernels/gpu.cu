#include "data_types.h"
#include "stdio.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>

namespace gpu {

    std::string PrevFile{};
    int PrevLine{0};
    void checkErr(const std::string &file, int line) {
      cudaError_t Error = cudaGetLastError();
      if (Error != cudaSuccess) {
        std::cout << std::endl << file << ", line " << line
                  << ": " << cudaGetErrorString(Error) << " (" << Error << ")" << std::endl;
        if (PrevLine > 0)
          std::cout << "Previous CUDA call:" << std::endl
                    << PrevFile << ", line " << PrevLine << std::endl;
        throw;
      }
      PrevFile = file;
      PrevLine = line;
    }
    #define CHECK_ERR checkErr(__FILE__,__LINE__)

    
    __global__ void kernel_computeMandelbrotSet(uint32_t* field, ProgramSettins settings) {
        const size_t linIndex = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t FIELD_SIZE = settings.numPixelsX * settings.numPixelsY;
        
        constexpr real MAX_VALUE = 4;
        const real deltaX = ((settings.domain.maxX - settings.domain.minX) / (real)settings.numPixelsX);
        const real deltaY = ((settings.domain.maxY - settings.domain.minY) / (real)settings.numPixelsY);

        if (linIndex < FIELD_SIZE) {

            real x0 = (linIndex % settings.numPixelsX) * deltaX + settings.domain.minX;
            real y0 = (linIndex / settings.numPixelsX) * deltaY + settings.domain.minY;

            real absValue = 0.0;
            real x = 0.0;
            real y = 0.0;

            uint32_t iteration = 0;
                
            while ((absValue < MAX_VALUE) && (iteration < settings.maxIterations)) {
                real tmp = x * x  - y * y  + x0;
                y = 2 * x * y + y0;
                x = tmp;
                absValue = x * x + y * y; 
                ++iteration;
            }

            field[linIndex] = iteration;
        }
    }

    __global__ void kernel_computeMandelbrotSet(uint32_t* field) {
        const size_t linIndex = threadIdx.x + blockIdx.x * blockDim.x;
        field[linIndex] = 50;
    }

    void computeMandelbrotSet(std::vector<uint32_t>& inputField, const ProgramSettins& settings, double& computeTime) {
        uint32_t *defField{nullptr};
        const size_t FIELD_SIZE = inputField.size();
        
        cudaMalloc(&defField, FIELD_SIZE * sizeof(uint32_t)); CHECK_ERR;
        cudaMemcpy(defField, inputField.data(), FIELD_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice); CHECK_ERR;
        
        
        constexpr unsigned BLOCK_SIZE = 512;
        const unsigned NUM_BLOCKS = (BLOCK_SIZE + FIELD_SIZE - 1) / BLOCK_SIZE;

        std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();

        kernel_computeMandelbrotSet<<<NUM_BLOCKS, BLOCK_SIZE>>>(defField, settings); CHECK_ERR;
        cudaDeviceSynchronize(); CHECK_ERR;

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        computeTime = (std::chrono::duration_cast<std::chrono::duration<double>>(end - begin)).count();
        
        
        cudaMemcpy(const_cast<uint32_t*>(inputField.data()), defField, FIELD_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost); CHECK_ERR;

        cudaFree(defField); CHECK_ERR;
    }

    void printHardwareInfo() {
        int currentDeviceId{-1};
        cudaGetDevice(&currentDeviceId); CHECK_ERR;
        
        cudaDeviceProp Property;
        cudaGetDeviceProperties(&Property, currentDeviceId); CHECK_ERR;

        std::ostringstream info;
        info << "Name: " << Property.name << '\n'
             << "totalGlobalMem: " << Property.totalGlobalMem << '\n'
             << "sharedMemPerBlock: " << Property.sharedMemPerBlock << '\n'
             << "regsPerBlock: " << Property.regsPerBlock << '\n'
             << "warpSize: " << Property.warpSize << '\n'
             << "memPitch: " << Property.memPitch << '\n'
             << "maxThreadsPerBlock: " << Property.maxThreadsPerBlock << '\n'
             << "totalConstMem: " << Property.totalConstMem << '\n'
             << "clockRate: " << Property.clockRate << '\n'
             << "multiProcessorCount: " << Property.multiProcessorCount << '\n'
             << "integrated: " << Property.integrated << '\n'
             << "canMapHostMemory: " << Property.canMapHostMemory << '\n'
             << "computeMode: " << Property.computeMode << '\n'
             << "concurrentKernels: " << Property.concurrentKernels << '\n'
             << "pciBusID: " << Property.pciBusID << '\n'
             << "pciDeviceID: " << Property.pciDeviceID << '\n';

        std::cout << info.str() << std::endl;
    }
}