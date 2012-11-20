#include <cstdlib>
#include <cmath>

#include "fractal.h"
#include "common.h"

// Calculate if c is in Mandelbrot set.
// Return number of iterations.
inline ElementType mandelbrot(ElementType c_r, ElementType c_i, IterationType iterations)
{  
   ElementType z_r = c_r;
   ElementType z_i = c_i;
   
   ElementType z2_r = z_r * z_r;
   ElementType z2_i = z_i * z_i;
   
   IterationType n = 0;
   
   while(n < iterations && z2_r + z2_i < 4.0)
   {           
      z_i = 2.0 * z_r * z_i + c_i;
      z_r = z2_r - z2_i + c_r;
   
      z2_r = z_r * z_r;
      z2_i = z_i * z_i;
      
      n++;
   }

   z_i = 2.0 * z_r * z_i + c_i;
   z_r = z2_r - z2_i + c_r;
   
   z2_r = z_r * z_r;
   z2_i = z_i * z_i;

   z_i = 2.0 * z_r * z_i + c_i;
   z_r = z2_r - z2_i + c_r;
   
   z2_r = z_r * z_r;
   z2_i = z_i * z_i;
      
   n += 2;

   if(n > iterations)
   {
		return (ElementType)iterations;
   }
   else
   {
      return (ElementType)n + 1.0 - log(log(sqrt(z2_r + z2_i)))/log(2.0);;
   }
}

ElementType ssaaPixel(DimensionSqType index, ElementType yMax, ElementType xMin,
                               ElementType xScale, ElementType yScale, IterationType iterations,
                               DimensionType width, AlisingFactorType ssaafactor)
{
   DimensionType dx = index % width;
   DimensionType dy = index / width;

   ElementType c_i = yMax - (ElementType)dy * yScale;
   ElementType c_r = xMin + (ElementType)dx * xScale;

   // Get sub pixel width and height
   ElementType xSubScale = xScale / ((ElementType)ssaafactor);
   ElementType ySubScale = yScale / ((ElementType)ssaafactor);
   
   // Get the centre of the top left subpixel
   ElementType xSubMin = c_r - (xScale / 2.0) + (xSubScale / 2.0);
   ElementType ySubMax = c_i + (yScale / 2.0) - (ySubScale / 2.0);

   AlisingFactorSqType factor2 = (AlisingFactorSqType)ssaafactor * (AlisingFactorSqType)ssaafactor;
   
   // Get the values for each pixel in fractal   
   ElementType subpixels[MAX_ALIASING_FACTOR * MAX_ALIASING_FACTOR];
   
   for(AlisingFactorType x = 0; x < ssaafactor; x++)
   {
      for(AlisingFactorType y = 0; y < ssaafactor; y++)
      {
         subpixels[x * ssaafactor + y] = mandelbrot(xSubMin + xSubScale * x, ySubMax - ySubScale * y, iterations);
      }
   }

   if(factor2 % 2 != 0)
   {
      return median(subpixels, factor2 / 2,  factor2);
   }
   else
   {
      ElementType rtn = median(subpixels, factor2 / 2 - 1,  factor2) + median(subpixels, factor2 / 2,  factor2);
      return rtn / 2.0;
   }
}

void antiAliasingSSAA(ElementType* image, DimensionSqType width, DimensionSqType height,
                           AlisingFactorType ssaaFactor, IterationType iterations,
                           ElementType yMax, ElementType xMin,
                           ElementType xScale, ElementType yScale)
{
   DimensionSqType c = 0;
   IterationType nInt = (IterationType)image[c];
   
   // Top left corner
   if(nInt != (IterationType)image[c + 1] ||
      nInt != (IterationType)image[c + height])
   {
      image[c] = ssaaPixel(c, yMax, xMin,
                           xScale, yScale, iterations,
                           width, ssaaFactor);
   }
   
   // Bottom left corner
   c = width * (height - 1);
   nInt = (IterationType)image[c];
   if(nInt != (IterationType)image[c + 1] ||
      nInt != (IterationType)image[c - width])
   {
      image[c] = ssaaPixel(c, yMax, xMin,
                           xScale, yScale, iterations,
                           width, ssaaFactor);
   }
   
   // Top right corner
   c = width - 1;
   nInt = (IterationType)image[c];
   if(nInt != (IterationType)image[c - 1] ||
      nInt != (IterationType)image[c + height])
   {
      image[c] = ssaaPixel(c, yMax, xMin,
                           xScale, yScale, iterations,
                           width, ssaaFactor);
   }
   
   // Bottom right corner
   c = width * height - 1;
   nInt = (IterationType)image[c];
   if(nInt != (IterationType)image[c - 1] ||
      nInt != (IterationType)image[c - width])
   {
      image[c] = ssaaPixel(c, yMax, xMin,
                           xScale, yScale, iterations,
                           width, ssaaFactor);
   }
      
   //Top border
   for(DimensionSqType x = 1; x < width - 1; x++)
   {
      c = x;
      nInt = (IterationType)image[x];
         
      if(nInt != (IterationType)image[x - 1] ||
         nInt != (IterationType)image[x + 1] ||
         nInt != (IterationType)image[x + width])
      {
         image[c] = ssaaPixel(c, yMax, xMin,
                           xScale, yScale, iterations,
                           width, ssaaFactor);                         
      }
   }
      
   //Left border
   for(DimensionSqType y = 1; y < height - 1; y++)
   {
      c = y * width;
      nInt = (IterationType)image[c];
         
      if(nInt != (IterationType)image[c + 1] ||
         nInt != (IterationType)image[c - width] ||
         nInt != (IterationType)image[c + width])
      {
         image[c] = ssaaPixel(c, yMax, xMin,
                           xScale, yScale, iterations,
                           width, ssaaFactor);
      }
   }
      
   //Bottom border
   for(DimensionSqType x = 1; x < width - 1; x++)
   {
      c = width * (height - 1) + x;
      nInt = (IterationType)image[c];
         
      if(nInt != (IterationType)image[c - 1] ||
         nInt != (IterationType)image[c + 1] || 
         nInt != (IterationType)image[c - width])
      {
         image[c] = ssaaPixel(c, yMax, xMin,
                           xScale, yScale, iterations,
                           width, ssaaFactor);
      }
   }
      
   //Right border
   for(DimensionSqType y = 1; y < height - 1; y++)
   {
      c = y * width - 1;
      nInt = (IterationType)image[c];
         
      if(nInt != (IterationType)image[c - 1] ||
         nInt != (IterationType)image[c - width] ||
         nInt != (IterationType)image[c + width])
      {
         image[c] = ssaaPixel(c, yMax, xMin,
                           xScale, yScale, iterations,
                           width, ssaaFactor);
      }
   }
      
   // Middle
   for(DimensionSqType y = 1; y < height - 1; y++)
   {      
      for(DimensionSqType x = 1; x < width - 1; x++)
      {
         c = width * y + x;
         nInt = (IterationType)image[c];
            
         if(nInt != (IterationType)image[c - 1] ||
            nInt != (IterationType)image[c + 1] ||
            nInt != (IterationType)image[c - width] ||
            nInt != (IterationType)image[c + width])
         {
            image[c] = ssaaPixel(c, yMax, xMin,
                           xScale, yScale, iterations,
                           width, ssaaFactor);
         }
      }
   }
}

void fractal(ElementType* image, DimensionType imgWidth, DimensionType imgHeight,
                IterationType iterations, ElementType xMin, ElementType xMax,
                ElementType yMin, ElementType yMax, AlisingFactorType ssaaFactor)
{
   // Cast things to a size that prevents many casts later
   ElementType c_i;
   ElementType c_r;

   // Save from casting many times
   DimensionSqType width = (DimensionSqType)imgWidth;
   DimensionSqType height = (DimensionSqType)imgHeight;
   
   // Get width and height of pixel
   ElementType xScale = (xMax - xMin) / ((ElementType)width);
   ElementType yScale = (yMax - yMin) / ((ElementType)height);

   DimensionSqType c;

   // Get the values for each pixel in fractal
   for(DimensionSqType y = 0; y < height; y++)
   {
      c_i = yMax - (ElementType)y * yScale;
      c = y * width;

      for(DimensionSqType x = 0; x < width; x++)
      {
         c_r = xMin + (ElementType)x * xScale;

         image[c + x] = mandelbrot(c_r, c_i, iterations);
      }
   }

   //Anti-alising
   if(ssaaFactor > 1)
   {
      antiAliasingSSAA(image, width, height, ssaaFactor, iterations, yMax, xMin, xScale, yScale);
   }
}
