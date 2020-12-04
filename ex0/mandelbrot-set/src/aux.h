#ifndef AUX_H
#define AUX_H

#include "data_types.h"
#include <vector>

ProgramSettins getInputFromCmd(int argc, char *argv[]);
std::string modeToString(MODE mode);
void checkInputData(const ProgramSettins &settings);
std::ostream &operator<<(std::ostream &stream, const ProgramSettins &data);
void writeData(const std::vector<unsigned char> &outputField, const ProgramSettins &settings);

#endif // AUX_H