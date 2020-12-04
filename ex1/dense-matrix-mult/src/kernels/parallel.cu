#include "kernels.h"
#include "util.hpp"
#include <iostream>

namespace gpu {
size_t get1DGrid(size_t blockSize, size_t matrixSize) {
    // TODO: complete function
    return 1;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_matrixMultGlobal(const float *devA, const float *devB, float *devC,
                                        const int size) {
  int row = threadIdx.X;
  int column = threadIdx.y; 

  if(row < size && column < size){

    // matrix are store colun-wise ? (col1, col2, ..)
      float acc = 0.0f;

      for (j = 0; j<size; j++){
          acc += devA[row + size*j] * devB[column*size + j];
      }

      devC[row + size* column] += acc;

  }
}

void executeMatrixMultGlobal(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                             const Configuration &config) {
    for (int i = 0; i < config.numRepeats; ++i) {
        kernel_matrixMultGlobal<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void kernel_matrixMultTiled(const float *__restrict__ devA,
                                       const float *__restrict__ devB, float *__restrict__ devC,
                                       const size_t size) {
  // TODO: complete function
}


void executeMatrixMultTiled(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                            const Configuration &config) {
    switch (config.tileSize) {
        case 4:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<4><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 8:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<8><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 16:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<16><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 32:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<32><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void kernel_matrixMultCoalesced(const float *__restrict__ devA,
                                           const float *__restrict__ devB, float *__restrict__ devC,
                                           const size_t size) {
  // TODO: complete function
}


void executeMatrixMultCoalesced(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                                const Configuration &config) {
    switch (config.tileSize) {
        case 4:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<4><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 8:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<8><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 16:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<16>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 32:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<32>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_matrixMultCoalescedDym(const float *__restrict__ devA,
                                              const float *__restrict__ devB,
                                              float *__restrict__ devC, const size_t size) {
  // TODO: complete function
}


void executeMatrixMultCoalescedDym(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                                   const Configuration &config) {
    const size_t shrMemSize = 2 * config.tileSize * config.tileSize * sizeof(float);
    for (int i = 0; i < config.numRepeats; ++i) {
        kernel_matrixMultCoalescedDym<<<dimGrid, dimBlock, shrMemSize>>>(Ad, Bd, Cd,
                                                                         config.matrixSize);
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void kernel_matrixMultOverlapped(const float *__restrict__ devA,
                                            const float *__restrict__ devB,
                                            float *__restrict__ devC, const size_t size) {
    // TODO: complete function
}


void executeMatrixMultOverlapped(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                                 const Configuration &config) {
    switch (config.tileSize) {
        case 4:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<4>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 8:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<8>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 16:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<16>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 32:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<32>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
    }
    CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
void matrixMult(float *Ad, float *Bd, float *Cd, const Configuration &config) {
    // TODO: adjust dimBlock and dimGrid
    dim3 dimBlock(1, 1);
    const size_t Grid1D = get1DGrid(1, 1);
    dim3 dimGrid(1, 1);

    switch (config.kernelType) {
        case KernelType::KERNEL_GLOBAL:
            executeMatrixMultGlobal(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_TILED:
            executeMatrixMultTiled(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_COALESCED:
            executeMatrixMultCoalesced(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_COALESCED_DYM:
            executeMatrixMultCoalescedDym(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_OVERLAPPED:
            executeMatrixMultOverlapped(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
    }
    CHECK_ERR;
}
} // namespace gpu