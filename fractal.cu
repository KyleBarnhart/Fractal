#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "fractal.h"
#include "common.h"

__device__ __constant__ float LOG_2 = 0.69314718056;

__device__ void swap(ElementType* a, ElementType* b)
{
   ElementType t = *a;
   *a = *b;
   *b = t;
}

__device__ ElementType getMedian(ElementType* arr, AlisingFactorSqType median, AlisingFactorSqType n) 
{
    AlisingFactorSqType low, high ;
    AlisingFactorSqType middle, ll, hh;

    low = 0 ; high = n-1 ;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                swap(&arr[low], &arr[high]) ;
            return arr[median];
        }

       /* Find median of low, middle and high items; swap into position low */
       middle = (low + high) / 2;
       if (arr[middle] > arr[high])    swap(&arr[middle], &arr[high]) ;
       if (arr[low] > arr[high])       swap(&arr[low], &arr[high]) ;
       if (arr[middle] > arr[low])     swap(&arr[middle], &arr[low]) ;

       /* Swap low item (now in position middle) into position (low+1) */
       swap(&arr[middle], &arr[low+1]) ;

       /* Nibble from each end towards middle, swapping items when stuck */
       ll = low + 1;
       hh = high;
       for (;;) {
           do ll++; while (arr[low] > arr[ll]) ;
           do hh--; while (arr[hh]  > arr[low]) ;

           if (hh < ll)
           break;

           swap(&arr[ll], &arr[hh]) ;
       }

       /* Swap middle item (in position low) back into correct position */
       swap(&arr[low], &arr[hh]) ;

       /* Re-set active partition */
       if (hh <= median)
           low = ll;
       if (hh >= median)
           high = hh - 1;
    }
}

__device__ ElementType mandelbrot(ElementType c_i, ElementType c_r, IterationType iterations)
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
      return (ElementType)n + 1.0 - __logf(__logf(__dsqrt_rn(z2_r + z2_i)))/LOG_2;
   }
}

// Calculate if c is in Mandelbrot set.
// Return number of iterations.
__global__ void getFractal(ElementType* img, ElementType yMax, ElementType xMin, ElementType xScale, ElementType yScale, IterationType iterations, DimensionType width, DimensionType height)
{  
   DimensionType dx = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
   DimensionType dy = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

   if(dx >= width || dy >= height)
		return;
   
   // This is fine because so few registers are used
   img[(DimensionSqType)dy * (DimensionSqType)width + (DimensionSqType)dx] = mandelbrot(yMax - (ElementType)dy * yScale,
                                                                                        xMin + (ElementType)dx * xScale,
                                                                                        iterations);
}

// Calculate if c is in Mandelbrot set.
// Return number of iterations.
__global__ void getFractalSSAA(ElementType* img, DimensionSqType* list, DimensionSqType length, ElementType yMax, ElementType xMin,
                               ElementType xScale, ElementType yScale, IterationType iterations,
                               DimensionType width, AlisingFactorType ssaafactor)
{  
   DimensionType curr = blockIdx.x * BLOCK_SIZE_SSAA + threadIdx.x;

   if(curr >= length)
		return;

   DimensionSqType val = list[curr];
   
   ElementType xSubScale = xScale / ((ElementType)ssaafactor);
   ElementType ySubScale = yScale / ((ElementType)ssaafactor);

   // Get the centre of the top left subpixel
   xMin = xMin + (ElementType)(val % width) * xScale - (xScale / 2.0) + (xSubScale / 2.0);
   yMax = yMax - (ElementType)(val / width) * yScale + (yScale / 2.0) - (ySubScale / 2.0);
   
   // Get the values for each pixel in fractal   
   ElementType subpixels[MAX_ALIASING_FACTOR * MAX_ALIASING_FACTOR];
   
   for(AlisingFactorType x = 0; x < ssaafactor; x++)
   {
      for(AlisingFactorType y = 0; y < ssaafactor; y++)
      {
         subpixels[x * ssaafactor + y] = mandelbrot(yMax - ySubScale * y , xMin + xSubScale * x, iterations);
      }
   }

   AlisingFactorSqType factor2 = (AlisingFactorSqType)ssaafactor * (AlisingFactorSqType)ssaafactor;

   if(factor2 % 2 != 0)
   {
      img[val] = getMedian(subpixels, (AlisingFactorSqType)ssaafactor * (AlisingFactorSqType)ssaafactor / 2,  factor2);
   }
   else
   {
      img[val] = (getMedian(subpixels, factor2 / 2 - 1,  factor2)
                           + getMedian(subpixels, factor2 / 2,  factor2))
                         / 2.0;
   }
}

void antiAliasingSSAA(ElementType* image, ElementType* deviceImg, DimensionType width, DimensionType height,
                           AlisingFactorType ssaaFactor, IterationType iterations,
                           ElementType yMax, ElementType xMin,
                           ElementType xScale, ElementType yScale)
{
   DimensionSqType* ssaaMap = NULL;
   ssaaMap = (DimensionSqType*) malloc((DimensionSqType)width * (DimensionSqType)height * (DimensionSqType)sizeof(ElementType));

   DimensionSqType c = 0;
   IterationType nInt = (IterationType)image[c];
   DimensionSqType counter = 0;

   // Anti-alias all side because large images are tiled and we cannot detect what the other tile will be
   //Corners
   ssaaMap[0] = 0;
   ssaaMap[1] = width * (height - 1);
   ssaaMap[2] = width - 1;
   ssaaMap[3] = width * height - 1;
   
   counter = 4;

   //Top border and bottom border
   for(DimensionSqType x = 1; x < width - 1; x++)
   {
      ssaaMap[counter] = x;
      ssaaMap[counter+1] = width * (height - 1) + x;
      counter += 2;                          
   }
      
   //Left border and right border
   for(DimensionSqType y = 1; y < height - 1; y++)
   {
      ssaaMap[counter] = y * width;
      ssaaMap[counter+1] =  y * width - 1;
      counter += 2;
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
            ssaaMap[counter] = c;
            ++counter;
         }
      }
   }

   ssaaMap = (DimensionSqType*) realloc(ssaaMap, counter * sizeof(DimensionSqType));

   cudaError_t error;

   DimensionSqType arraySize = (DimensionSqType)width * (DimensionSqType)height * (DimensionSqType)sizeof(ElementType);
   
   // Get the values for each pixel in fractal   
   DimensionSqType* ssaaMapDevice;
   safeCudaMalloc((void**)&ssaaMapDevice, counter * sizeof(DimensionSqType));

   safeCudaMemcpy(ssaaMapDevice, ssaaMap, counter * sizeof(DimensionSqType), cudaMemcpyHostToDevice);

   free(ssaaMap);
   
   unsigned blocks = (counter / BLOCK_SIZE_SSAA) + ((counter % BLOCK_SIZE_SSAA == 0) ? 0 : 1);
   getFractalSSAA<<<blocks, BLOCK_SIZE_SSAA>>>(deviceImg, ssaaMapDevice, counter, yMax, xMin,
                               xScale, yScale, iterations,
                               width, ssaaFactor);
   if ((error = cudaGetLastError()) != cudaSuccess)
		displayCudeError(error);

   // Get fractal values from GPU
   safeCudaMemcpy(image, deviceImg, arraySize, cudaMemcpyDeviceToHost);
   
   safeCudaFree(ssaaMapDevice);
}

void fractal(ElementType* image, DimensionType width, DimensionType height,
                IterationType iterations, ElementType xMin, ElementType xMax,
                ElementType yMin, ElementType yMax, AlisingFactorType ssaaFactor)
{
   cudaError_t error;
   
   // Get width and height of pixel
   ElementType xScale = (xMax - xMin) / ((ElementType)width);
   ElementType yScale = (yMax - yMin) / ((ElementType)height);

   DimensionSqType arraySize = (DimensionSqType)width * (DimensionSqType)height * (DimensionSqType)sizeof(ElementType);

   // Array of floats for the GPU
   ElementType* deviceImg;
   safeCudaMalloc((void**)&deviceImg, arraySize);

   safeCudaMemset(deviceImg, 0, arraySize);

   // Run fractal on GPU
   int gridWidth = (width / BLOCK_SIZE_X) + (width % BLOCK_SIZE_X > 0 ? 1 : 0);
   int gridHeight =  (height / BLOCK_SIZE_Y) + (height % BLOCK_SIZE_Y > 0 ? 1 : 0);

   dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
   dim3 dimGrid(gridWidth, gridHeight);
   getFractal<<<dimGrid, dimBlock>>>(deviceImg, yMax, xMin, xScale, yScale, iterations, width, height);
   if ((error = cudaGetLastError()) != cudaSuccess)
		displayCudeError(error);

   // Get fractal values from GPU
   safeCudaMemcpy(image, deviceImg, arraySize, cudaMemcpyDeviceToHost);

   if(ssaaFactor > 1)
   {
      antiAliasingSSAA(image, deviceImg, width, height, ssaaFactor, iterations, yMax, xMin, xScale, yScale);
   }

   safeCudaFree(deviceImg);
}
