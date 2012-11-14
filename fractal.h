#ifndef FRACTAL_H
#define FRACTAL_H

#include "common.h"

void fractal(ElementType* image, DimensionType width, DimensionType height,
                IterationType iterations, ElementType xMin, ElementType xMax,
                ElementType yMin, ElementType yMax, AlisingFactorType ssaaFactor = 0);

#endif
