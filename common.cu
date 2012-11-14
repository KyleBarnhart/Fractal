/*
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipes in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 *  This code by Nicolas Devillard - 1998. Public domain.
 *
 * http://ndevilla.free.fr/median/median/index.html
 */
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "common.h"

__device__ void getRGB(ElementType value, BYTE* rgb)
{ 
   short colourInt = (short)(value * 1791.0f);
   
   BYTE bracket = colourInt / 256;
   BYTE colour = (BYTE)(colourInt % 256);

   switch (bracket)
   {
      case 0:
         rgb[0] = colour;
         rgb[1] = 0;
         rgb[2] = 0;
         break;
          
      case 1:
         rgb[0] = 255;
         rgb[1] = colour;
         rgb[2] = 0;
         break;
          
      case 2:
         rgb[0] = 255 - colour;
         rgb[1] = 255;
         rgb[2] = 0;
         break;

      case 3:
         rgb[0] = 0;
         rgb[1] = 255;
         rgb[2] = colour;
          break;

      case 4:
         rgb[0] = 0;
         rgb[1] = 255 - colour;
         rgb[2] = 255;
         break;

      case 5:
         rgb[0] = colour;
         rgb[1] = 0;
         rgb[2] = 255;
         break;

      case 6:
         rgb[0] = 255 - colour;
         rgb[1] = 0;
         rgb[2] = 255 - colour;
         break;

      default:
         rgb[0] = 0;
         rgb[1] = 0;
         rgb[2] = 0;
          break;    
   }
}

__global__ void getBmpRGB(BYTE* image, ElementType* values, DimensionType width, DimensionType height, IterationType iterations)
{
   DimensionType dy = blockIdx.y * BLOCK_SIZE + threadIdx.y;  
   DimensionType dx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
   
   if(dx >= width || dy >= height)
      return; 

   DimensionType c = dy * width + dx;
   
   BYTE rgbValue[3];

   getRGB(values[c]/(ElementType)iterations, rgbValue);
      
   image[c*3]      = rgbValue[2];
   image[c*3 + 1]  = rgbValue[1];
   image[c*3 + 2]  = rgbValue[0];
}

__global__ void getBmpRGBfromHistorgram(ElementType* map, BYTE* image, ElementType* values, DimensionType width, DimensionType height)
{
   DimensionType dy = blockIdx.y * BLOCK_SIZE + threadIdx.y;  
   DimensionType dx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
   
   if(dx >= width || dy >= height)
      return; 

   DimensionType c = dy * width + dx;

   IterationType ival = (IterationType)values[c];

   ElementType colourVal = map[ival] + (values[c] - (ElementType)ival) * (map[ival + 1] - map[ival]);
   
   BYTE rgbValue[3];

   getRGB(colourVal, rgbValue);
      
   image[c*3]      = rgbValue[2];
   image[c*3 + 1]  = rgbValue[1];
   image[c*3 + 2]  = rgbValue[0];
}

int displayCudeError(cudaError_t error) {
	std::cerr << cudaGetErrorString(error) << std::endl;
	std::cin.ignore();
	exit((int)error);
}

void histogramToColourMap(DimensionSqType* histogram, ElementType* map, IterationType iterations, DimensionSqType resolution)
{
   ElementType res = (ElementType)resolution;
      
   map[0] = 0.0;

   // Map colors to pixels based on the histogram
   for(IterationType i = 1; i < iterations + 1; i++)
   {
      if (histogram[i] == 0)
         map[i] = map[i-1];
      else
         map[i] = map[i-1] + (ElementType)histogram[i] / res;
   }
}

void valueToRGB(ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height)
{
   cudaError_t error;
   
   DimensionSqType resolution = (DimensionSqType)width * (DimensionSqType)height;

   // Bytes
   BYTE* deviceBytes;
   error = cudaMalloc((void**)&deviceBytes, resolution * 3);
   if(error != cudaSuccess)
		displayCudeError(error);

   // Array of floats for the GPU
   ElementType* deviceValues;
   error = cudaMalloc((void**)&deviceValues, resolution * (DimensionSqType)sizeof(ElementType));
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaMemcpy(deviceValues, values, resolution * (DimensionSqType)sizeof(ElementType), cudaMemcpyHostToDevice);
   if(error != cudaSuccess)
		displayCudeError(error);

   // Run fractal on GPU
   int gridWidth = (width / BLOCK_SIZE) + (width % BLOCK_SIZE > 0 ? 1 : 0);
   int gridHeight =  (height / BLOCK_SIZE) + (height % BLOCK_SIZE > 0 ? 1 : 0);

   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   dim3 dimGrid(gridWidth, gridHeight);

   getBmpRGB<<<dimGrid, dimBlock>>>(deviceBytes, deviceValues, width, height, iterations);
   if ((error = cudaGetLastError()) != cudaSuccess)
		displayCudeError(error);

   error = cudaFree(deviceValues);
   if(error != cudaSuccess)
		displayCudeError(error);

   // Get fractal values from GPU
   error = cudaMemcpy(image, deviceBytes, resolution * 3, cudaMemcpyDeviceToHost);
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaFree(deviceBytes);
   if(error != cudaSuccess)
		displayCudeError(error);
}

void mapValueToRGB(ElementType* map, ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height)
{
   cudaError_t error;
   
   DimensionSqType resolution = (DimensionSqType)width * (DimensionSqType)height;

   // Map
   ElementType* deviceMapValues;
   error = cudaMalloc((void**)&deviceMapValues, (iterations + 1) * sizeof(ElementType));
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaMemcpy(deviceMapValues, map, (iterations + 1) * sizeof(ElementType), cudaMemcpyHostToDevice);
   if(error != cudaSuccess)
		displayCudeError(error);

   // Bytes
   BYTE* deviceBytes;
   error = cudaMalloc((void**)&deviceBytes, resolution * 3);
   if(error != cudaSuccess)
		displayCudeError(error);

   // Array of floats for the GPU
   ElementType* deviceValues;
   error = cudaMalloc((void**)&deviceValues, resolution * (DimensionSqType)sizeof(ElementType));
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaMemcpy(deviceValues, values, resolution * (DimensionSqType)sizeof(ElementType), cudaMemcpyHostToDevice);
   if(error != cudaSuccess)
		displayCudeError(error);

   // Run fractal on GPU
   int gridWidth = (width / BLOCK_SIZE) + (width % BLOCK_SIZE > 0 ? 1 : 0);
   int gridHeight =  (height / BLOCK_SIZE) + (height % BLOCK_SIZE > 0 ? 1 : 0);

   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   dim3 dimGrid(gridWidth, gridHeight);

   getBmpRGBfromHistorgram<<<dimGrid, dimBlock>>>(deviceMapValues, deviceBytes, deviceValues, width, height);
   if ((error = cudaGetLastError()) != cudaSuccess)
		displayCudeError(error);

   error = cudaFree(deviceValues);
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaFree(deviceMapValues);
   if(error != cudaSuccess)
		displayCudeError(error);

   // Get fractal values from GPU
   error = cudaMemcpy(image, deviceBytes, resolution * 3, cudaMemcpyDeviceToHost);
   if(error != cudaSuccess)
		displayCudeError(error);

   error = cudaFree(deviceBytes);
   if(error != cudaSuccess)
		displayCudeError(error);
}