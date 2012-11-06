#include <cstdlib>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "fractal.h"
#include "common.h"

// Calculate if c is in Mandelbrot set.
// Return number of iterations.
__global__ void mandelbrot(float* img, float yMax, float xMin, float xScale, float yScale, unsigned iterations, unsigned width, unsigned height)
{  
   unsigned long c = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

   if(c >= height * width)
		return;

   unsigned y = c / width;
   unsigned x =  c % width;
   
   float c_i = yMax - (float)y * yScale;
   float c_r = xMin + (float)x * xScale;

   float z_r = c_r;
   float z_i = c_i;
   
   float z2_r = z_r * z_r;
   float z2_i = z_i * z_i;
   
   unsigned n = 0;
   
   while(n < iterations && z2_r + z2_i < 4.0f)
   {           
      z_i = 2.0f * z_r * z_i + c_i;
      z_r = z2_r - z2_i + c_r;
   
      z2_r = z_r * z_r;
      z2_i = z_i * z_i;
      
      n++;
   }

   z_i = 2.0f * z_r * z_i + c_i;
   z_r = z2_r - z2_i + c_r;
   
   z2_r = z_r * z_r;
   z2_i = z_i * z_i;

   z_i = 2.0f * z_r * z_i + c_i;
   z_r = z2_r - z2_i + c_r;
   
   z2_r = z_r * z_r;
   z2_i = z_i * z_i;
      
   n += 2;

   if(n > iterations)
   {
		img[c] = (float)iterations;
   }
   else
   {
      img[c] = (float)n - log(log(sqrt(z2_r + z2_i)))/log(2.0f);
   }
}

int displayCudeError(cudaError_t error) {
	std::cerr << cudaGetErrorString(error) << std::endl;
	std::cin.ignore();
	exit((int)error);
}

ElementType antiAliasingSSAA(AlisingFactorType factor, IterationType iterations,
                           ElementType x, ElementType y,
                           ElementType xScale, ElementType yScale)
{
   cudaError_t error;

   // Get sub pixel width and height
   ElementType xSubScale = xScale / ((ElementType)factor);
   ElementType ySubScale = yScale / ((ElementType)factor);
   
   // Get the centre of the top left subpixel
   ElementType xMin = x - (xScale / 2.0) + (xSubScale / 2.0);
   ElementType yMax = y + (yScale / 2.0) - (ySubScale / 2.0);

   AlisingFactorSqType factor2 = factor * factor;
   
   ElementType* n = NULL;
   n = (ElementType*) malloc(factor * factor * sizeof(ElementType));
   
   // Get the values for each pixel in fractal   
   ElementType* deviceImg;
   error = cudaMalloc((void**)&deviceImg, factor2 * sizeof(ElementType));
   if(error != cudaSuccess)
		displayCudeError(error);
   
   mandelbrot<<<1, factor2>>>(deviceImg, yMax, xMin, xSubScale, ySubScale, iterations, factor, factor);
   if ((error = cudaGetLastError()) != cudaSuccess)
		displayCudeError(error);

   error = cudaMemcpy(n, deviceImg, factor2 * sizeof(ElementType), cudaMemcpyDeviceToHost);
   if(error != cudaSuccess)
		displayCudeError(error);
   
   error = cudaFree(deviceImg);
   if(error != cudaSuccess)
		displayCudeError(error);

   ElementType result = median(n, factor2);
   
   free(n);

   return result;
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
   cudaError_t error;

   // Cast things to a size that prevents many casts later
   DimensionSqType width = imgWidth;
   DimensionSqType height = imgHeight;
   DimensionSqType resolution = width * height;
   
   // Get width and height of pixel
   ElementType xScale = (xMax - xMin) / ((ElementType)width);
   ElementType yScale = (yMax - yMin) / ((ElementType)height);
   
   // Array of floats for the GPU
   ElementType* deviceImg;
   error = cudaMalloc((void**)&deviceImg, resolution * sizeof(ElementType));
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaMemset(deviceImg, 0, resolution * sizeof(ElementType));
   if(error != cudaSuccess)
		displayCudeError(error);

   // Run fractal on GPU
   CudaIndexType blocks = resolution / BLOCK_WIDTH + (resolution % BLOCK_WIDTH > 0 ? 1 : 0);
   mandelbrot<<<blocks, BLOCK_WIDTH>>>(deviceImg, yMax, xMin, xScale, yScale, iterations, width, height);
   if ((error = cudaGetLastError()) != cudaSuccess)
		displayCudeError(error);

   // Contains actual results
   ElementType* imgValues = NULL;
   imgValues = (ElementType*) malloc (resolution * sizeof(ElementType));
   if (imgValues == NULL)
      exit (1);
   
   // Get fractal values from GPU
   error = cudaMemcpy(imgValues, deviceImg, resolution * sizeof(ElementType), cudaMemcpyDeviceToHost);
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaFree(deviceImg);
   if(error != cudaSuccess)
		displayCudeError(error);

   // Used to compute edges and histogram
   unsigned long* nValues = NULL;
   nValues = (unsigned long*) malloc (resolution * sizeof(unsigned long));
   if (nValues == NULL)
      exit (2);

   // Get integer values of fractal floats
   for(unsigned long i = 0; i < resolution; i++)
   {
      nValues[i] = (unsigned long)imgValues[i];
   }

   //Anti-alising
   if(ssaaFactor > 1)
   {
      IterationType nInt;
      DimensionSqType c;
      
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
