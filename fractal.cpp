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
   
   while(n < iterations && z2_r + z2_i < 4)
   {           
      z_i = 2.0 * z_r * z_i + c_i;
      z_r = z2_r - z2_i + c_r;
   
      z2_r = z_r * z_r;
      z2_i = z_i * z_i;
      
      n++;
   }
   
   // Iterate 2 more times to prevent errors
   z_i = 2.0 * z_r * z_i + c_i;
   z_r = z2_r - z2_i + c_r;

   z2_r = z_r * z_r;
   z2_i = z_i * z_i;

   z_i = 2.0 * z_r * z_i + c_i;
   z_r = z2_r - z2_i + c_r;

   z2_r = z_r * z_r;
   z2_i = z_i * z_i;
   
   n += 2;

   if (n > iterations)
	   return iterations;

   return (ElementType)n - log(log(sqrt(z2_r + z2_i)))/log(2.0);
}

ElementType antiAliasingSSAA(AlisingFactorType factor, IterationType iterations,
                           ElementType x, ElementType y,
                           ElementType xScale, ElementType yScale)
{
   ElementType c_i, c_r;

   // Get sub pixel width and height
   ElementType xSubScale = xScale / ((ElementType)factor);
   ElementType ySubScale = yScale / ((ElementType)factor);
   
   // Get the centre of the top left subpixel
   ElementType xMin = x - (xScale / 2.0) + (xSubScale / 2.0);
   ElementType yMax = y + (yScale / 2.0) - (ySubScale / 2.0);

   AlisingFactorSqType factor2 = factor * factor;
   
   ElementType* n = NULL;
   n = (ElementType*) malloc(factor * factor * sizeof(ElementType));
   
   /* The x,y is the centre.
	* Divide the pixel into factor^2 subpixels and find centre of each.
	*/
   for(AlisingFactorSqType y = 0; y < factor; y++)
   {
      c_i = yMax - (ElementType)y * ySubScale;
      
      for(AlisingFactorSqType x = 0; x < factor; x++)
      {
         c_r = xMin + (ElementType)x * xSubScale;
         
         n[y * factor + x] = mandelbrot(c_r, c_i, iterations);
      }
   }
   
   ElementType rtnValue = median(n, factor2);
   
   free(n);
   
   return rtnValue;
}

void fractal(BYTE* image, DimensionType imgWidth, DimensionType imgHeight,
                IterationType iterations, ElementType xMin, ElementType xMax,
                ElementType yMin, ElementType yMax)
{
   fractal(image, imgWidth, imgHeight,
                iterations, xMin, xMax,
                yMin, yMax, 0);
}

void fractal(BYTE* image, DimensionType imgWidth, DimensionType imgHeight,
                IterationType iterations, ElementType xMin, ElementType xMax,
                ElementType yMin, ElementType yMax, AlisingFactorType ssaaFactor)
{
   // Cast things to a size that prevents many casts later
   ElementType c_i;
   ElementType c_r;
   DimensionSqType width = imgWidth;
   DimensionSqType height = imgHeight;
   DimensionSqType resolution = width * height;
   
   // Get width and height of pixel
   ElementType xScale = (xMax - xMin) / ((ElementType)width);
   ElementType yScale = (yMax - yMin) / ((ElementType)height);
   
   // Used to make colour change smoother
   IterationType* nValues;
   nValues = (IterationType*) malloc ((uint64_t)resolution * (uint64_t)sizeof(IterationType));
   if (nValues == NULL)
      exit (1);
   
   ElementType* imgValues;
   imgValues = (ElementType*) malloc ((uint64_t)resolution * (uint64_t)sizeof(ElementType));
   if (imgValues == NULL)
      exit (2);

   ElementType n;
   DimensionSqType c;

   // Get the values for each pixel in fractal
   for(DimensionSqType y = 0; y < height; y++)
   {
      c_i = yMax - (ElementType)y * yScale;
      c = y * width;

      for(DimensionSqType x = 0; x < width; x++)
      {
         c_r = xMin + (ElementType)x * xScale;

         n = mandelbrot(c_r, c_i, iterations);

         imgValues[c + x] = n;

         nValues[c + x] = (IterationType)n;
      }
   }

   //Anti-alising
   if(ssaaFactor > 1)
   {
      IterationType nInt;
      
      //Corners
      c = 0;
      nInt = nValues[c];
      if(nValues[c + 1] != nInt ||
         nValues[c + height] != nInt)
      {
         imgValues[c] = antiAliasingSSAA(ssaaFactor, iterations,
								           xMin, yMax,
                                   xScale, yScale);
      }
      
      c = width * (height - 1);
      nInt = nValues[c];
      if(nValues[c + 1] != nInt ||
         nValues[c - width] != nInt)
      {
         imgValues[c] = antiAliasingSSAA(ssaaFactor, iterations,
                                   xMin, yMin,
                                   xScale, yScale);
      }
      
      c = width - 1;
      nInt = nValues[c];
      if(nValues[c - 1] != nInt ||
         nValues[c + height] != nInt)
      {
         imgValues[c] = antiAliasingSSAA(ssaaFactor, iterations,
                                   xMax, yMax,
                                   xScale, yScale);
      }
      
      c = width * height - 1;
      nInt = nValues[c];
      if(nValues[c - 1] != nInt ||
         nValues[c - width] != nInt)
      {
         imgValues[c] = antiAliasingSSAA(ssaaFactor, iterations,
                                   xMax, yMin,
                                   xScale, yScale);
      }
      
      //Top border
      for(DimensionSqType x = 1; x < width - 1; x++)
      {
         nInt = nValues[x];
         
         if(nValues[x - 1] != nInt ||
            nValues[x + 1] != nInt ||
            nValues[x + width] != nInt)
         {
            imgValues[x] = antiAliasingSSAA(ssaaFactor, iterations,
                                      xMin + x * xScale, yMax,
                                      xScale, yScale);                          
         }
      }
      
      //Left border
      for(DimensionSqType y = 1; y < height - 1; y++)
      {
         c = y * width;
         nInt = nValues[c];
         
         if(nValues[c + 1] != nInt ||
            nValues[c - width] != nInt ||
            nValues[c + width] != nInt)
         {
            imgValues[c] = antiAliasingSSAA(ssaaFactor, iterations,
                                      xMin, yMax - y * yScale,
                                      xScale, yScale);
         }
      }
      
      //Bottom border
      for(DimensionSqType x = 1; x < width - 1; x++)
      {
         c = width * (height - 1) + x;
         nInt = nValues[c];
         
         if(nValues[c - 1] != nInt ||
            nValues[c + 1] != nInt || 
            nValues[c - width] != nInt)
         {
            imgValues[c] = antiAliasingSSAA(ssaaFactor, iterations,
                                      xMin + x * xScale, yMin,
                                      xScale, yScale);
         }
      }
      
      //Right border
      for(DimensionSqType y = 1; y < height - 1; y++)
      {
         c = y * width - 1;
         nInt = nValues[c];
         
         if(nValues[c - 1] != nInt ||
            nValues[c - width] != nInt ||
            nValues[c + width] != nInt)
         {
            imgValues[c] = antiAliasingSSAA(ssaaFactor, iterations,
                                      xMax, yMax - y * yScale,
                                      xScale, yScale);
         }
      }
      
      // Middle
      for(DimensionSqType y = 1; y < height - 1; y++)
      {      
         for(DimensionSqType x = 1; x < width - 1; x++)
         {
            c = width * y + x;
            nInt = nValues[c];
            
            if(nValues[c - 1] != nInt ||
               nValues[c + 1] != nInt ||
               nValues[c - width] != nInt ||
               nValues[c + width] != nInt)
            {
               imgValues[c] = antiAliasingSSAA(ssaaFactor, iterations,
                                         xMin + x * xScale, yMax - y * yScale,
                                         xScale, yScale);
            }
         }
      }
   }
   
   // Make histogram
   DimensionSqType* histogram = NULL;
   histogram = (DimensionSqType*) calloc(iterations + 1, sizeof(DimensionSqType));
   if (histogram == NULL)
      exit (3);

   for(DimensionSqType i = 0; i < height * width; i++)
   {  
      histogram[nValues[i]]++;
   }
   
   // Used to map colours to pixels
   ElementType* map = NULL;
   map = (ElementType*) malloc((iterations + 1) * sizeof(ElementType));
   if (map == NULL)
      exit (4);
      
   map[0] = 0.0;

   // Map colors to pixels based on the histogram
   for(IterationType i = 1; i < iterations + 1; i++)
   {
      if (histogram[i] == 0)
         map[i] = map[i-1];
      else
         map[i] = map[i-1] + (ElementType)histogram[i] / (ElementType)resolution;
   }

   free(histogram);
   
   ElementType val;
   ElementType colourVal;
   IterationType ival;
   DimensionSqType position = 0;
   
   BYTE* rgbValue;
   rgbValue = (BYTE*) malloc(3);

   for(DimensionSqType i = 0; i < resolution; i++)
   {
      val = imgValues[i];
      ival = nValues[i];

      colourVal = 1.0;
      if (ival < iterations)
         colourVal = map[ival] + (val - (ElementType)ival) * (map[ival + 1] - map[ival]);
         
      getRGB(colourVal, rgbValue);
      
      image[i*3]      = rgbValue[0];
      image[i*3 + 1]  = rgbValue[1];
      image[i*3 + 2]  = rgbValue[2];
      
      position += 3;
   }
   
   free(rgbValue);
   free(map);
   free(imgValues);
   free(nValues);
}
