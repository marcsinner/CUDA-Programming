#ifndef KERNELS_H
#define KERNELS_H

#include "data_types.h"

namespace serial {
    void computeMandelbrotSet(std::vector<uint32_t>& inputField, const ProgramSettins& settings);
}

namespace threading {
    void computeMandelbrotSet(std::vector<uint32_t>& inputField, const ProgramSettins& settings);
}

namespace gpu {
    void computeMandelbrotSet(std::vector<uint32_t>& inputField, const ProgramSettins& settings, double& computeTime);
    void printHardwareInfo();
}

#endif  // KERNELS_H