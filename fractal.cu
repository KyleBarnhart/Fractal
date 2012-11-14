#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "fractal.h"
#include "common.h"

__device__ void swap(ElementType* a, ElementType* b)
{
   ElementType t = *a;
   *a = *b;
   *b = t;
}

__device__ ElementType getMedian(ElementType* arr, AlisingFactorSqType n) 
{
    AlisingFactorSqType low, high ;
    AlisingFactorSqType median;
    AlisingFactorSqType middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            return (arr[low] + arr[high]) / 2 ;
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
      return (ElementType)n + 1.0 - log(log(sqrt(z2_r + z2_i)))/log(2.0);;
   }
}

// Calculate if c is in Mandelbrot set.
// Return number of iterations.
__global__ void getFractal(ElementType* img, ElementType yMax, ElementType xMin, ElementType xScale, ElementType yScale, IterationType iterations, DimensionType width, DimensionType height)
{  
   DimensionType dx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
   DimensionType dy = blockIdx.y * BLOCK_SIZE + threadIdx.y;

   if(dx >= width || dy >= height)
		return;
   
   ElementType c_i = yMax - (ElementType)dy * yScale;
   ElementType c_r = xMin + (ElementType)dx * xScale;

   DimensionSqType c = (DimensionSqType)dy * (DimensionSqType)width + (DimensionSqType)dx;
   img[c] = mandelbrot(c_i, c_r, iterations);
}

// Calculate if c is in Mandelbrot set.
// Return number of iterations.
__global__ void getFractalSSAA(ElementType* img, DimensionSqType* list, DimensionSqType length, ElementType yMax, ElementType xMin,
                               ElementType xScale, ElementType yScale, IterationType iterations,
                               DimensionType width, AlisingFactorType ssaafactor)
{  
   DimensionType curr = blockIdx.x * BLOCK_SIZE * BLOCK_SIZE + threadIdx.x;

   if(curr >= length)
		return;

   DimensionType dx = list[curr] % width;
   DimensionType dy = list[curr] / width;

   ElementType c_i = yMax - (ElementType)dy * yScale;
   ElementType c_r = xMin + (ElementType)dx * xScale;

   // Get sub pixel width and height
   ElementType xSubScale = xScale / ((ElementType)ssaafactor);
   ElementType ySubScale = yScale / ((ElementType)ssaafactor);
   
   // Get the centre of the top left subpixel
   xMin = c_r - (xScale / 2.0) + (xSubScale / 2.0);
   yMax = c_i + (yScale / 2.0) - (ySubScale / 2.0);

   AlisingFactorSqType factor2 = (AlisingFactorSqType)ssaafactor * (AlisingFactorSqType)ssaafactor;
   
   // Get the values for each pixel in fractal   
   ElementType subpixels[MAX_ALIASING_FACTOR * MAX_ALIASING_FACTOR];
   
   for(AlisingFactorType x = 0; x < ssaafactor; x++)
   {
      for(AlisingFactorType y = 0; y < ssaafactor; y++)
      {
         subpixels[x * ssaafactor + y] = mandelbrot(yMax - ySubScale * y , xMin + xSubScale * x, iterations);
      }
   }

   img[list[curr]] = getMedian(subpixels, factor2);
}

void antiAliasingSSAA(ElementType* image, DimensionType width, DimensionType height,
                           AlisingFactorType ssaaFactor, IterationType iterations,
                           ElementType yMax, ElementType xMin,
                           ElementType xScale, ElementType yScale)
{
   DimensionSqType* ssaaMap = NULL;
   ssaaMap = (DimensionSqType*) malloc((DimensionSqType)width * (DimensionSqType)height * (DimensionSqType)sizeof(ElementType));

   DimensionSqType c = 0;
   IterationType nInt = (IterationType)image[c];
   DimensionSqType counter = 0;

   //Corners
   if(nInt != (IterationType)image[c + 1] ||
      nInt != (IterationType)image[c + height])
   {
      ssaaMap[counter] = c;
      ++counter;
   }
      
   c = width * (height - 1);
   nInt = (IterationType)image[c];
   if(nInt != (IterationType)image[c + 1] ||
      nInt != (IterationType)image[c - width])
   {
      ssaaMap[counter] = c;
      ++counter;
   }
      
   c = width - 1;
   nInt = (IterationType)image[c];
   if(nInt != (IterationType)image[c - 1] ||
      nInt != (IterationType)image[c + height])
   {
      ssaaMap[counter] = c;
      ++counter;
   }
      
   c = width * height - 1;
   nInt = (IterationType)image[c];
   if(nInt != (IterationType)image[c - 1] ||
      nInt != (IterationType)image[c - width])
   {
      ssaaMap[counter] = c;
      ++counter;
   }
      
   //Top border
   for(DimensionSqType x = 1; x < width - 1; x++)
   {
      nInt = (IterationType)image[x];
         
      if(nInt != (IterationType)image[x - 1] ||
         nInt != (IterationType)image[x + 1] ||
         nInt != (IterationType)image[x + width])
      {
         ssaaMap[counter] = c;
         ++counter;                          
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
         ssaaMap[counter] = c;
         ++counter;
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
         ssaaMap[counter] = c;
         ++counter;
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
         ssaaMap[counter] = c;
         ++counter;
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
   error = cudaMalloc((void**)&ssaaMapDevice, counter * sizeof(DimensionSqType));
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaMemcpy(ssaaMapDevice, ssaaMap, counter * sizeof(DimensionSqType), cudaMemcpyHostToDevice);
   if(error != cudaSuccess)
		displayCudeError(error);

   free(ssaaMap);

   // Array of floats for the GPU
   ElementType* deviceImg;
   error = cudaMalloc((void**)&deviceImg, arraySize);
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaMemcpy(deviceImg, image, arraySize, cudaMemcpyHostToDevice);
   if(error != cudaSuccess)
		displayCudeError(error);
   
   unsigned blocks = (counter / (BLOCK_SIZE * BLOCK_SIZE)) + ((counter % (BLOCK_SIZE * BLOCK_SIZE) == 0) ? 0 : 1);
   getFractalSSAA<<<blocks, BLOCK_SIZE * BLOCK_SIZE>>>(deviceImg, ssaaMapDevice, counter, yMax, xMin,
                               xScale, yScale, iterations,
                               width, ssaaFactor);
   if ((error = cudaGetLastError()) != cudaSuccess)
		displayCudeError(error);

   // Get fractal values from GPU
   error = cudaMemcpy(image, deviceImg, arraySize, cudaMemcpyDeviceToHost);
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaFree(deviceImg);
   if(error != cudaSuccess)
		displayCudeError(error);
   
   error = cudaFree(ssaaMapDevice);
   if(error != cudaSuccess)
		displayCudeError(error);
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
   error = cudaMalloc((void**)&deviceImg, arraySize);
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaMemset(deviceImg, 0, arraySize);
   if(error != cudaSuccess)
		displayCudeError(error);

   // Run fractal on GPU
   int gridWidth = (width / BLOCK_SIZE) + (width % BLOCK_SIZE > 0 ? 1 : 0);
   int gridHeight =  (height / BLOCK_SIZE) + (height % BLOCK_SIZE > 0 ? 1 : 0);

   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   dim3 dimGrid(gridWidth, gridHeight);

   getFractal<<<dimGrid, dimBlock>>>(deviceImg, yMax, xMin, xScale, yScale, iterations, width, height);
   if ((error = cudaGetLastError()) != cudaSuccess)
		displayCudeError(error);

   // Get fractal values from GPU
   error = cudaMemcpy(image, deviceImg, arraySize, cudaMemcpyDeviceToHost);
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaFree(deviceImg);
   if(error != cudaSuccess)
		displayCudeError(error);

   if(ssaaFactor > 1)
   {
      antiAliasingSSAA(image, width, height, ssaaFactor, iterations, yMax, xMin, xScale, yScale);
   }
}
