#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <cmath>

#include "fractal.h"
#include "bmp.h"

using namespace std;

int main(int argc, char* argv[])
{                      
   double zoomFactor = 1.0;
   double x = -0.5;
   double y = 0.0;
   unsigned ssaa = 0;
   unsigned width = 800;
   unsigned height = 600;
   unsigned iterations = 50;
   double endZoom = 0.0;
   unsigned frames = 1;

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
         zoomFactor = atof(argv[i + 1]);
         i += 2;
      }
      
      // x coordinate
      else if (strcmp(argv[i],"-x") == 0 && argc >= i + 1)
      {
         x = atof(argv[i + 1]);
         i += 2;
      }
      
      // y coordinate
      else if (strcmp(argv[i],"-y") == 0 && argc >= i + 1)
      {
         y = atof(argv[i + 1]);
         i += 2;
      }
      
      // Anti-alising (selective supersampiling)
      else if (strcmp(argv[i],"-ssaa") == 0 && argc >= i + 1)
      {
         ssaa = atoi(argv[i + 1]);
         i += 2;
      }
      
      // Animated zoom
      else if (strcmp(argv[i],"-animate") == 0 && argc >= i + 2)
      { 
         endZoom = atof(argv[i + 1]);
         frames = strtoul(argv[i + 2], NULL, 0);
         i += 3;        
      }
      
      // Default
      else
      {
         cout << "Usage is [-res <image width> <image height>]"
                    "[-i <iterations>] [-z <zoom factor>]"
                    "[-x <x coord>] [-y <y coord>]"
                    "[-ssaa <antialising factor>]"
                    "[-animate <end zoom factor> <frames>]\n";
         exit(0);
      }
   }
   
   double rSize = 3.0 / zoomFactor;
   double iSize = 3.0 * (double)height / (double)width / zoomFactor;
   
   double rMin = x - rSize / 2.0;
   double rMax = x + rSize / 2.0;
   double iMin = y - iSize / 2.0;
   double iMax = y + iSize / 2.0;

   unsigned char* image = NULL;
   char* filename;
   filename = (char*)malloc(13 * sizeof(char));
   strcpy(filename, "img00001.bmp");
   
   image = fractal(width, height, iterations, rMin, rMax, iMin, iMax, ssaa);
   
   saveAsBmp(image, width, height, filename);
   
   if (frames > 1)
   {
      double zoomMultiplyer = pow(endZoom - zoomFactor, 1.0/frames);
      
      for (unsigned i = 2; i <= frames; i++)
      {         
         rSize /= zoomMultiplyer;
         iSize /= zoomMultiplyer;
         
         rMin = x - rSize / 2.0;
         rMax = x + rSize / 2.0;
         iMin = y - iSize / 2.0;
         iMax = y + iSize / 2.0;
         
         sprintf (filename, "img%05u.bmp", i);

         free(image);       
         
         image = fractal(width, height, iterations, rMin, rMax, iMin, iMax, ssaa);
         saveAsBmp(image, width, height, filename);
      }
   }
   
   free(filename);
   free(image);
}
