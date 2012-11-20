/*
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipes in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 *  This code by Nicolas Devillard - 1998. Public domain.
 *
 * http://ndevilla.free.fr/median/median/index.html
 */
#include "common.h"

#define ELEM_SWAP(a,b) { register ElementType t=(a);(a)=(b);(b)=t; }

ElementType median(ElementType* arr, AlisingFactorSqType median, AlisingFactorSqType n) 
{
    AlisingFactorSqType low, high ;
    AlisingFactorSqType middle, ll, hh;

    low = 0 ; high = n-1 ;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[median];
        }

       /* Find median of low, middle and high items; swap into position low */
       middle = (low + high) / 2;
       if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
       if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
       if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

       /* Swap low item (now in position middle) into position (low+1) */
       ELEM_SWAP(arr[middle], arr[low+1]) ;

       /* Nibble from each end towards middle, swapping items when stuck */
       ll = low + 1;
       hh = high;
       for (;;) {
           do ll++; while (arr[low] > arr[ll]) ;
           do hh--; while (arr[hh]  > arr[low]) ;

           if (hh < ll)
           break;

           ELEM_SWAP(arr[ll], arr[hh]) ;
       }

       /* Swap middle item (in position low) back into correct position */
       ELEM_SWAP(arr[low], arr[hh]) ;

       /* Re-set active partition */
       if (hh <= median)
           low = ll;
       if (hh >= median)
           high = hh - 1;
    }
}

#undef ELEM_SWAP

void getRGB(ElementType value, BYTE rgb[])
{ 
   short colourInt = (short)(value * 1791.0);
   
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

void histogramToColourMap(DimensionSqType* histogram, ElementType* map, IterationType iterations, DimensionSqType resolution)
{
   ElementType res = (ElementType)resolution;
      
   map[0] = 0.0;

   // Map colors to pixels based on the histogram
   for(IterationType i = 1; i < iterations + 1; i++)
   {
      map[i] = map[i-1] + histogram[i] / res;
   }
}

void valueToRGB(ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height)
{ 
   DimensionSqType resolution = (DimensionSqType)width * (DimensionSqType)height;

   BYTE rgbValue[3];

   // Bytes
   for(IterationType i = 0; i < iterations; ++i)
   {
      getRGB(values[i]/(ElementType)iterations, rgbValue);
      
      image[i*3]      = rgbValue[2];
      image[i*3 + 1]  = rgbValue[1];
      image[i*3 + 2]  = rgbValue[0];
   }
}

void mapValueToRGB(ElementType* map, ElementType* values, BYTE* image, IterationType iterations, DimensionType width, DimensionType height)
{   
   DimensionSqType resolution = (DimensionSqType)width * (DimensionSqType)height;

   BYTE rgbValue[3];
   IterationType ival;
   ElementType colourVal;

   // Bytes
   for(DimensionSqType i = 0; i < resolution; ++i)
   {
      ival = (IterationType)values[i];

      colourVal = map[ival] + (values[i] - (ElementType)ival) * (map[ival + 1] - map[ival]);

      getRGB(colourVal, rgbValue);
      
      image[i*3]      = rgbValue[2];
      image[i*3 + 1]  = rgbValue[1];
      image[i*3 + 2]  = rgbValue[0];
   }
}