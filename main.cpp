#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <cmath>

#include "common.h"
#include "fractal.h"
#include "bmp.h"

using namespace std;

int main(int argc, char* argv[])
{                      
   ElementType zoomFactor = 100.0;
   ElementType x = 0.001643721969;//-0.5;
   ElementType y = -0.8224676332991;//0.0;
   AlisingFactorType ssaa = 0;
   DimensionType width = 19200;
   DimensionType height = 10800;
   IterationType iterations = 1200;
   ElementType endZoom = 1.0;
   FrameType frames = 1;

   int i = 1;
   while(i < argc)
   {
      // Image resolution
      if (strcmp(argv[i],"-res") == 0 && argc >= i + 2)
      { 
         width = atoi(argv[i + 1]);
         height = atoi(argv[i + 2]);
         i += 3;      
      }
      
      // Iterations
      else if (strcmp(argv[i],"-i") == 0 && argc >= i + 1)
      {
         iterations = atoi(argv[i + 1]);
         i += 2;
      }
      
      // Zoom factor
      else if (strcmp(argv[i],"-z") == 0 && argc >= i + 1)
      {
         zoomFactor = (ElementType)atof(argv[i + 1]);
         i += 2;
      }
      
      // x coordinate
      else if (strcmp(argv[i],"-x") == 0 && argc >= i + 1)
      {
         x = (ElementType)atof(argv[i + 1]);
         i += 2;
      }
      
      // y coordinate
      else if (strcmp(argv[i],"-y") == 0 && argc >= i + 1)
      {
         y = (ElementType)atof(argv[i + 1]);
         i += 2;
      }
      
      // Anti-alising (selective supersampiling)
      else if (strcmp(argv[i],"-ssaa") == 0 && argc >= i + 1)
      {
         ssaa = atoi(argv[i + 1]);
         i += 2;
      }
      
      // Animated zoom
      else if (strcmp(argv[i],"-sequence") == 0 && argc >= i + 2)
      { 
         endZoom = (ElementType)atof(argv[i + 1]);
         frames = atoi(argv[i + 2]);
         i += 3;        
      }
      
      // Default
      else
      {
         cout << "Usage is [-res <image width> <image height>]"
                    "[-i <iterations>] [-z <zoom factor>]"
                    "[-x <x coord>] [-y <y coord>]"
                    "[-ssaa <antialising factor>]"
                    "[-sequence <end zoom factor> <frames>]\n";
         exit(0);
      }
   }

   if (ssaa > MAX_ALIASING_FACTOR)
   {
      cout << "Anti-alaising factor is too hight.";
      exit(0);
   }
   
   ElementType rSize = 3.0 / zoomFactor;
   ElementType iSize = 3.0 * (ElementType)height / (ElementType)width / zoomFactor;
   
   DimensionSqType resolution = (DimensionSqType)width * (DimensionSqType)height;

   DimensionType rowsPerPass = (resolution < MAX_PIXELS_PER_PASS) ? height : MAX_PIXELS_PER_PASS / width;
   DimensionType passes = (height / rowsPerPass) + ((height % rowsPerPass == 0) ? 0 : 1);

   ElementType zoomMultiplyer = 1.0;
   if (endZoom > zoomFactor)
      zoomMultiplyer = pow(endZoom/zoomFactor, (ElementType)1.0/frames);
      
   for (FrameType f = 1; f < frames + 1; f++)
   {                  
      ElementType rMin = x - rSize / 2.0;
      ElementType rMax = x + rSize / 2.0;
      ElementType iMin = y - iSize / 2.0;
      ElementType iMax = y + iSize / 2.0;

      DimensionType passHeight = rowsPerPass;
      ElementType pass_iSize = iSize * ((ElementType)passHeight / (ElementType)height);

      ElementType pass_iMax = iMax;
      ElementType pass_iMin = iMax - pass_iSize;

      // Get histogram
      DimensionSqType* histogram = NULL;
      histogram = (DimensionSqType*) malloc ((iterations + 1) * sizeof(DimensionSqType));
      if (histogram == NULL)
         exit (1);

      memset(histogram, 0, (iterations + 1) * sizeof(DimensionSqType));

      // Get fractal portion and save
      for(DimensionType i = 0; i < passes; i++)
      {
         if(pass_iMin < iMin)
         {
            passHeight = height - rowsPerPass * i;
            pass_iMin = iMin;
         }

         // Get fractal values
         ElementType* nValues;
         nValues = (ElementType*) malloc (passHeight * width * sizeof(ElementType));
         if (nValues == NULL)
               exit (2);

         fractal(nValues, width, passHeight, iterations, rMin, rMax, pass_iMin, pass_iMax, 0);

         // Make histogram
         for(DimensionSqType j = 0; j < passHeight * width; ++j)
         {
            histogram[(DimensionSqType)nValues[j]]++;
         }

         free(nValues);

         pass_iMax -= pass_iSize;
         pass_iMin -= pass_iSize;
      }

      // Reset
      passHeight = rowsPerPass;
      pass_iMax = iMax;
      pass_iMin = iMax - pass_iSize;

      // Get colour map
      // Used to map colours to pixels
      ElementType* map = NULL;
      map = (ElementType*) malloc((iterations + 1) * sizeof(ElementType));
      if (map == NULL)
         exit (3);

      histogramToColourMap(histogram, map, iterations, resolution);

      free(histogram);

      // Create file
      char* filename;
      filename = (char*)malloc(40 * sizeof(char));
      if (filename == NULL)
            exit (4);
      
      sprintf (filename, "img%05u.bmp", f);

      startBmp(width, height, filename);

      // Get fractal portion and save
      for(DimensionType i = 0; i < passes; i++)
      {
         if(pass_iMin < iMin)
         {
            passHeight = height - rowsPerPass * i;
            pass_iMin = iMin;
         }

         // Get fractal values
         ElementType* nValues;
         nValues = (ElementType*) malloc (passHeight * width * sizeof(ElementType));
         if (nValues == NULL)
               exit (5);

         fractal(nValues, width, passHeight, iterations, rMin, rMax, pass_iMin, pass_iMax, ssaa);

         // Map values to rgb
         BYTE* image;
         image = (BYTE*) malloc (passHeight * width * 3); 
         if (image == NULL)
               exit (6);

         mapValueToRGB(map, nValues, image, iterations, width, passHeight);

         free(nValues);

         // Save image
         appendBmp(image, width, passHeight, filename);

         free(image);

         pass_iMax -= pass_iSize;
         pass_iMin -= pass_iSize;
      }

      free(map);
      free(filename);

      rSize /= zoomMultiplyer;
      iSize /= zoomMultiplyer;
   }
}