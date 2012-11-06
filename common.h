#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <climits>

typedef uint8_t BYTE;
typedef uint16_t WORD;
typedef uint32_t DWORD;
typedef int32_t SDWORD;
typedef uint64_t ULONG;

typedef uint16_t IterationType;
typedef float ElementType;
typedef uint16_t DimensionType;
typedef uint32_t DimensionSqType;
typedef uint8_t AlisingFactorType;
typedef uint8_t AlisingFactorSqType;
typedef uint16_t FrameType;

const uint32_t MAX_RESOLUTION = 4294967295;
const uint8_t MAX_ALIASING_FACTOR = 16;

// CUDA
typedef uint32_t CudaIndexType;
const CudaIndexType BLOCK_WIDTH = 256;

ElementType median(ElementType* arr, AlisingFactorSqType n);

void getRGB(ElementType value, BYTE rgb[]);

#endif
