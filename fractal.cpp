#include <cstdlib>
#include <cmath>

#include "fractal.h"
#include "common.h"

// Calculate if c is in Mandelbrot set.
// Return number of iterations.
inline double mandelbrot(double c_r, double c_i, unsigned long iterations)
{  
   double z_r = c_r;
   double z_i = c_i;
   
   double z2_r = z_r * z_r;
   double z2_i = z_i * z_i;
   
   unsigned long n = 0;
   
   while(n < iterations && z2_r + z2_i < 4)
   {           
      z_i = 2 * z_r * z_i + c_i;
      z_r = z2_r - z2_i + c_r;
   
      z2_r = z_r * z_r;
      z2_i = z_i * z_i;
      
      n++;
   }
   
   // Iterate 2 more times to prevent errors
   z_i = 2 * z_r * z_i + c_i;
   z_r = z2_r - z2_i + c_r;

   z2_r = z_r * z_r;
   z2_i = z_i * z_i;

   z_i = 2 * z_r * z_i + c_i;
   z_r = z2_r - z2_i + c_r;

   z2_r = z_r * z_r;
   z2_i = z_i * z_i;
   
   n += 2; 
   
   if (n < iterations + 2)
      return n - log(log(sqrt(z2_r + z2_i)))/log(2);
   
   return iterations;
}

double antiAliasingSSAA(unsigned factor, unsigned long iterations,
                           double xMin, double xMax,
                           double yMin, double yMax)
{
   double c_i, c_r;

   // Get width and height of pixel
   double xScale = (xMax - xMin) / ((double)factor);
   double yScale = (yMax - yMin) / ((double)factor);
   double halfxScale = xScale / 2.0;
   double halfyScale = yScale / 2.0;

   unsigned long factor2 = (unsigned long)factor * (unsigned long)factor;
   double n[factor2];
   
   for(unsigned y = 0; y < factor; y++)
   {
      c_i = yMax - (double)y * yScale - halfyScale;
      
      for(unsigned x = 0; x < factor; x++)
      {
         c_r = xMin + (double)x * xScale + halfxScale;
         
         n[y * factor + x] = mandelbrot(c_r, c_i, iterations);
      }
   }
   
   return median(n, factor2);
}

unsigned char* fractal(unsigned imgWidth, unsigned imgHeight,
                unsigned iterations, double xMin, double xMax,
                double yMin, double yMax)
{
   return fractal(imgWidth, imgHeight,
                iterations, xMin, xMax,
                yMin, yMax, 0);
}

unsigned char* fractal(unsigned imgWidth, unsigned imgHeight,
                unsigned iterations, double xMin, double xMax,
                double yMin, double yMax, unsigned ssaaFactor)
{
   // Cast things to a size that prevents many casts later
   double c_i, c_r;
   unsigned long width = imgWidth;
   unsigned long height = imgHeight;
   unsigned long resolution = width * height;
   
   // Get width and height of pixel
   
   double xScale = (xMax - xMin) / ((double)width);
   double yScale = (yMax - yMin) / ((double)height);
   double halfxScale = xScale / 2.0;
   double halfyScale = yScale / 2.0;
   
   // Used to make colour change smoother
   unsigned long* nValues = NULL;
   double* img = NULL;
   
   nValues = (unsigned long*) malloc (resolution * sizeof(unsigned long));
   if (nValues == NULL)
      exit (1);
   
   img = (double*) malloc (resolution * sizeof(double));
   if (img == NULL)
      exit (2);
   
   double n;
   unsigned long c;
   
   // Get the values for each pixel in fractal
   for(unsigned long y = 0; y < height; y++)
   {
      c_i = yMax - (double)y * yScale - halfyScale;
      
      for(unsigned long x = 0; x < width; x++)
      {
         c_r = xMin + (double)x * xScale + halfxScale;
         
         n = mandelbrot(c_r, c_i, iterations);
         
         c = y * width + x;
   
         img[c] = n;
         
         nValues[c] = (unsigned long)n;
      }
   }
   
   //Anti-alising
   if(ssaaFactor > 1)
   {
      unsigned long nInt;
      
      //Corners
      c = 0;
      nInt = nValues[c];
      if(nValues[c + 1] != nInt ||
         nValues[c + height] != nInt)
      {
         img[c] = antiAliasingSSAA(ssaaFactor, iterations,
                        xMin, xMin + xScale,
                       yMax - yScale, yMax);
      }
      
      c = width * (height - 1);
      nInt = nValues[c];
      if(nValues[c + 1] != nInt ||
         nValues[c - width] != nInt)
      {
         img[c] = antiAliasingSSAA(ssaaFactor, iterations,
                                    xMin, xMin + xScale,
                                    yMin, yMin + yScale);
      }
      
      c = width - 1;
      nInt = nValues[c];
      if(nValues[c - 1] != nInt ||
         nValues[c + height] != nInt)
      {
         img[c] = antiAliasingSSAA(ssaaFactor, iterations,
                                 xMax - xScale, xMax,
                                 yMax - yScale, yMax);
      }
      
      c = width * height - 1;
      nInt = nValues[c];
      if(nValues[c - 1] != nInt ||
         nValues[c - width] != nInt)
      {
         img[c] = antiAliasingSSAA(ssaaFactor,
                                             iterations,
                                             xMax - xScale, xMax,
                                             yMin, yMin + yScale);
      }
      
      //Top border
      for(unsigned long x = 1; x < width - 1; x++)
      {
         nInt = nValues[x];
         
         if(nValues[x - 1] != nInt ||
            nValues[x + 1] != nInt ||
            nValues[x + width] != nInt)
         {
            img[x] = antiAliasingSSAA(ssaaFactor, iterations,
                           xMin + x * xScale, xMin + (x+1) * xScale,
                           yMax - yScale, yMax);                           
         }
      }
      
      //Left border
      for(unsigned long y = 1; y < height - 1; y++)
      {
         c = y * width;
         nInt = nValues[c];
         
         if(nValues[c + 1] != nInt ||
            nValues[c - width] != nInt ||
            nValues[c + width] != nInt)
         {
            img[c] = antiAliasingSSAA(ssaaFactor, iterations,
                           xMin, xMin + xScale,
                           yMax - (y+1) * yScale, yMax - y * yScale);
         }
      }
      
      //Bottom border
      for(unsigned long x = 1; x < width - 1; x++)
      {
         c = width * (height - 1) + x;
         nInt = nValues[c];
         
         if(nValues[c - 1] != nInt ||
            nValues[c + 1] != nInt || 
            nValues[c - width] != nInt)
         {
            img[c] = antiAliasingSSAA(ssaaFactor, iterations,
                           xMin + x * xScale, xMin + (x+1) * xScale,
                           yMin, yMin + yScale);
         }
      }
      
      //Right border
      for(unsigned long y = 1; y < height - 1; y++)
      {
         c = y * width - 1;
         nInt = nValues[c];
         
         if(nValues[c - 1] != nInt ||
            nValues[c - width] != nInt ||
            nValues[c + width] != nInt)
         {
            img[c] = antiAliasingSSAA(ssaaFactor, iterations,
                           xMax - xScale, xMax,
                           yMax - (y+1) * yScale, yMax - y * yScale);
         }
      }
      
      // Middle
      for(unsigned long y = 1; y < height - 1; y++)
      {      
         for(unsigned long x = 1; x < width - 1; x++)
         {
            c = width * y + x;
            nInt = nValues[c];
            
            if(nValues[c - 1] != nInt ||
               nValues[c + 1] != nInt ||
               nValues[c - width] != nInt ||
               nValues[c + width] != nInt)
            {
               img[c] = antiAliasingSSAA(ssaaFactor, iterations,
                              xMin + x * xScale, xMin + (x+1) * xScale,
                              yMax - (y+1) * yScale, yMax - y * yScale);
            }
         }
      }
   }
   
   // Make histogram
   unsigned long* histogram = NULL;
   histogram = (unsigned long*) calloc(iterations + 1, sizeof(unsigned long));
   if (histogram == NULL)
      exit (3);

   for(unsigned long i = 0; i < height * width; i++)
   {  
      histogram[nValues[i]]++;
   }
   
   // Used to map colours to pixels
   double* map = NULL;
   map = (double*) malloc((iterations + 1) * sizeof(double));
   if (map == NULL)
      exit (4);
      
   map[0] = 0.0;
   
   // Map colors to pixels based on the histogram
   for(unsigned long i = 1; i < iterations + 1; i++)
   {
      if (histogram[i] == 0)
         map[i] = map[i-1];
      else
         map[i] = map[i-1] + (double)histogram[i] / (double)resolution;
   }
   
   free(histogram);

   unsigned char* image = NULL;
   image = (unsigned char*) malloc(resolution * 3);
   if (image == NULL)
      exit (5);
   
   double val, colourVal;
   unsigned long ival;
   unsigned long position = 0;
   unsigned char rgb[3];
   
   for(unsigned long i = 0; i < resolution; i++)
   {
      val = img[i];
      ival = nValues[i];
      
      colourVal = 1.0;
      
      if (ival < iterations)
         colourVal = map[ival] + (val - (double)ival) * (map[ival + 1] - map[ival]);
         
      getRGB(colourVal, rgb);
      
      image[position]      = rgb[2];
      image[position + 1]  = rgb[1];
      image[position + 2]  = rgb[0];
      
      position += 3;
   }
   
   free(map);
   free(img);
   free(nValues);
   
   return image;
}
