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
   ElementType zoomFactor = 1.0;
   ElementType x = -0.5;
   ElementType y = 0.0;
   AlisingFactorType ssaa = 2;
   DimensionType width = 1920;
   DimensionType height = 1080;
   IterationType iterations = 50;
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

   if (height * width > MAX_RESOLUTION)
   {
      cout << "Resolution is too high.";
      exit(0);
   }

   if (ssaa > MAX_ALIASING_FACTOR)
   {
      cout << "Anti-alaising factor is too hight.";
      exit(0);
   }
   
   ElementType rSize = 3.0f / zoomFactor;
   ElementType iSize = 3.0f * (ElementType)height / (ElementType)width / zoomFactor;
   
   ElementType rMin = x - rSize / 2.0f;
   ElementType rMax = x + rSize / 2.0f;
   ElementType iMin = y - iSize / 2.0f;
   ElementType iMax = y + iSize / 2.0f;

   BYTE* image;
   image = (BYTE*) malloc(3 * (DimensionSqType)height * (DimensionType)width);

   char* filename;
   filename = (char*)malloc(13 * sizeof(char));
   strcpy(filename, "img00001.bmp");
   
   fractal(image, width, height, iterations, rMin, rMax, iMin, iMax, ssaa);
   
   saveAsBmp(image, width, height, filename);
   
   if (frames > 1)
   {
      ElementType zoomMultiplyer = pow(endZoom - zoomFactor, (ElementType)1.0/frames);
      
      for (FrameType i = 2; i <= frames; i++)
      {         
         rSize /= zoomMultiplyer;
         iSize /= zoomMultiplyer;
         
         rMin = x - rSize / 2.0f;
         rMax = x + rSize / 2.0f;
         iMin = y - iSize / 2.0f;
         iMax = y + iSize / 2.0f;
         
         sprintf (filename, "img%05u.bmp", i);

         free(image);       
         
         fractal(image, width, height, iterations, rMin, rMax, iMin, iMax, ssaa);
         saveAsBmp(image, width, height, filename);
      }
   }
   free(image);
   free(filename);
   
}
