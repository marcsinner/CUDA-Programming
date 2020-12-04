#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <string>

#if REAL_SIZE == 8
typedef double real;
#elif REAL_SIZE == 4
typedef float real;
#else
#error REAL_SIZE not supported.
#endif

enum class MODE : char { Serial, Threading, GPU };

struct Domain {
    real minX{};
    real maxX{};
    real minY{};
    real maxY{};
};


struct ProgramSettins {
    int x{0};
    int y{0};
    Domain domain{};
    uint32_t maxIterations{100};
    size_t numPixelsX{10240};
    size_t numPixelsY{7680};
    unsigned imageQuality{100};
    MODE mode{MODE::Serial};
    std::string outputFileName{"./set.jpeg"};
};

#endif // DATA_TYPES_H