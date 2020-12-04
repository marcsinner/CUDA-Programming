#include "data_types.h"
#include <vector>

namespace threading {
    void computeMandelbrotSet(std::vector<uint32_t>& inputField, const ProgramSettins& settings) {

        const real MAX_VALUE = 4;

        const uint32_t MAX_ITERATIONS = settings.maxIterations;
        const real deltaX = ((settings.domain.maxX - settings.domain.minX) / (real)settings.numPixelsX);
        const real deltaY = ((settings.domain.maxY - settings.domain.minY) / (real)settings.numPixelsY);

        uint32_t (*field)[settings.numPixelsX] = reinterpret_cast<uint32_t (*)[settings.numPixelsX]>(const_cast<uint32_t*>(inputField.data()));

        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < settings.numPixelsY; ++i) {
            for (size_t j = 0; j < settings.numPixelsX; ++j) {

                real absValue = 0.0;
                real x0 = j * deltaX + settings.domain.minX;
                real y0 = i * deltaY + settings.domain.minY;

                real x = 0.0;
                real y = 0.0;

                uint32_t iteration = 0;
                
                while ((absValue < MAX_VALUE) && (iteration < MAX_ITERATIONS)) {
                    real tmp = x * x  - y * y  + x0;
                    y = 2 * x * y + y0;
                    x = tmp;
                    absValue = x * x + y * y; 
                    ++iteration;
                }

                field[i][j] = iteration;
            }
        }
    }
}