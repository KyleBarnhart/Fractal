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
typedef double ElementType;
typedef uint16_t DimensionType;
typedef uint32_t DimensionSqType;
typedef uint8_t AlisingFactorType;
typedef uint8_t AlisingFactorSqType;
typedef uint16_t FrameType;

const uint8_t MAX_ALIASING_FACTOR = 16;
const uint32_t MAX_PIXELS_PER_PASS = 67108864;

ElementType median(ElementType* arr, AlisingFactorSqType median, AlisingFactorSqType n);

void histogramToColourMap(DimensionSqType* histogram, ElementType* map, IterationType iterations, DimensionSqType resolution);
void valueToRGB(ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height);
void mapValueToRGB(ElementType* map, ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height);

#endif
