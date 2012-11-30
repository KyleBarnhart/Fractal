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
const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 8;
const int BLOCK_SIZE_SSAA = 256;
const int BLOCK_SIZE_RGB = 16;

const int MAX_GRID_SIZE_X = 65536;

const uint8_t MAX_ALIASING_FACTOR = 16;

void displayCudeError(cudaError_t error);
void safeCudaMalloc(void** devPtr, size_t size);
void safeCudaMemcpy(void* dist, const void* src, size_t size, cudaMemcpyKind kind);
void safeCudaMemset(void* devPtr, int vlaue, size_t count);
void safeCudaFree(void* devPt);

void histogramToColourMap(DimensionSqType* histogram, ElementType* map, IterationType iterations, DimensionSqType resolution);
void valueToRGB(ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height);
void mapValueToRGB(ElementType* map, ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height);

#endif
