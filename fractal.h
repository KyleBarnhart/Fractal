#ifndef FRACTAL_H
#define FRACTAL_H

unsigned char* fractal(unsigned imgWidth, unsigned imgHeight,
                unsigned iterations, double xMin, double xMax,
                double yMin, double yMax);

unsigned char* fractal(unsigned imgWidth, unsigned imgHeight,
                unsigned iterations, double xMin, double xMax,
                double yMin, double yMax, unsigned ssaaFactor = 0);

#endif
