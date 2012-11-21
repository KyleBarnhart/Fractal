#ifndef COMMON_H
#define COMMON_H

#include <cstdint>

typedef uint8_t BYTE;
typedef uint16_t WORD;
typedef uint32_t DWORD;
typedef int32_t SDWORD;
typedef uint64_t ULONG;

typedef uint16_t IterationType;
typedef double ElementType;
typedef uint32_t DimensionType;
typedef uint64_t DimensionSqType;
typedef uint8_t AlisingFactorType;
typedef uint8_t AlisingFactorSqType;
typedef uint16_t FrameType;

#include <cuda_runtime.h>

// CUDA
const int BLOCK_SIZE = 16;
const int MAX_GRID_SIZE_X = 65536;

const uint8_t MAX_ALIASING_FACTOR = 16;

int displayCudeError(cudaError_t error);

void histogramToColourMap(DimensionSqType* histogram, ElementType* map, IterationType iterations, DimensionSqType resolution);
void valueToRGB(ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height);
void mapValueToRGB(ElementType* map, ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height);

#endif
