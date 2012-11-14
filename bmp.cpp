#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "common.h"
#include "bmp.h"

BmpFileHeader::BmpFileHeader()
{
   identifier = 0x4D42;
   reserved1 = 0;
   reserved2 = 0;
   offset = 54;
}

void BmpFileHeader::ToBytes(BYTE* buffer)
{
   buffer[0]  = (BYTE) identifier;
   buffer[1]  = (BYTE) (identifier >> 8);
   buffer[2]  = (BYTE) (filesize);
   buffer[3]  = (BYTE) (filesize >> 8);
   buffer[4]  = (BYTE) (filesize >> 16);
   buffer[5]  = (BYTE) (filesize >> 24);
   buffer[6]  = (BYTE) (reserved1);
   buffer[7]  = (BYTE) (reserved1 >> 8);
   buffer[8]  = (BYTE) (reserved2);
   buffer[9]  = (BYTE) (reserved2 >> 8);
   buffer[10] = (BYTE) offset;
   buffer[11] = (BYTE) (offset >> 8);
   buffer[12] = (BYTE) (offset >> 16);
   buffer[13] = (BYTE) (offset >> 24);
}

BmpInfoHeader::BmpInfoHeader()
{
   infoHeaderSize = 40;
   numColourPlanes = 1;
   bitsPerPixel = 24;
   compressionMethod = 0;
   imageSize = 0;
   widthPixelsPerMeter = 3780;
   heightPixelsPerMeter = 3780;
   numColours = 0;
   numImportantColours = 0;
}

void BmpInfoHeader::ToBytes(BYTE* buffer)
{
   buffer[0]  = (BYTE) infoHeaderSize;
   buffer[1]  = (BYTE) (infoHeaderSize >> 8);
   buffer[2]  = (BYTE) (infoHeaderSize >> 16);
   buffer[3]  = (BYTE) (infoHeaderSize >> 24);
   buffer[4]  = (BYTE) width;
   buffer[5]  = (BYTE) (width >> 8);
   buffer[6]  = (BYTE) (width >> 16);
   buffer[7]  = (BYTE) (width >> 24);
   buffer[8]  = (BYTE) height;
   buffer[9]  = (BYTE) (height >> 8);
   buffer[10] = (BYTE) (height >> 16);
   buffer[11] = (BYTE) (height >> 24);
   buffer[12] = (BYTE) numColourPlanes;
   buffer[13] = (BYTE) (numColourPlanes >> 8);
   buffer[14] = (BYTE) bitsPerPixel;
   buffer[15] = (BYTE) (bitsPerPixel >> 8);
   buffer[16] = (BYTE) compressionMethod;
   buffer[17] = (BYTE) (compressionMethod >> 8);
   buffer[18] = (BYTE) (compressionMethod >> 16);
   buffer[19] = (BYTE) (compressionMethod >> 24);
   buffer[20] = (BYTE) imageSize;
   buffer[21] = (BYTE) (imageSize >> 8);
   buffer[22] = (BYTE) (imageSize >> 16);
   buffer[23] = (BYTE) (imageSize >> 24);
   buffer[24] = (BYTE) widthPixelsPerMeter;
   buffer[25] = (BYTE) (widthPixelsPerMeter >> 8);
   buffer[26] = (BYTE) (widthPixelsPerMeter >> 16);
   buffer[27] = (BYTE) (widthPixelsPerMeter >> 24);
   buffer[28] = (BYTE) heightPixelsPerMeter;
   buffer[29] = (BYTE) (heightPixelsPerMeter >> 8);
   buffer[30] = (BYTE) (heightPixelsPerMeter >> 16);
   buffer[31] = (BYTE) (heightPixelsPerMeter >> 24);
   buffer[32] = (BYTE) numColours;
   buffer[33] = (BYTE) (numColours >> 8);
   buffer[34] = (BYTE) (numColours >> 16);
   buffer[35] = (BYTE) (numColours >> 24);
   buffer[36] = (BYTE) numImportantColours;
   buffer[37] = (BYTE) (numImportantColours >> 8);
   buffer[38] = (BYTE) (numImportantColours >> 16);
   buffer[39] = (BYTE) (numImportantColours >> 24);
}

void startBmp(DimensionType width, DimensionType height,
               const char* filename)
{
   DimensionSqType resolution = width * height;

   BmpFileHeader fileHeader;
   fileHeader.filesize = 54 + (DWORD)resolution * 3
                         + (3 * (DWORD)width) % 4
                         * (DWORD)height;

   BmpInfoHeader infoHeader;
   infoHeader.width = (SDWORD)width;
   infoHeader.height = (SDWORD)height;
   infoHeader.imageSize = (DWORD)resolution * 3;

   FILE* file;
   file = fopen(filename, "wb");

   BYTE* buffer;
   buffer = (BYTE*) malloc(40);

   fileHeader.ToBytes(buffer);
   fwrite(buffer, 1, 14, file);

   infoHeader.ToBytes(buffer);
   fwrite(buffer, 1, 40, file);

   free(buffer);

   fclose(file);
}

void appendBmp(BYTE* image, DimensionType width, DimensionType height,
               const char* filename)
{
   FILE* file;
   file = fopen(filename, "ab");

   // Padding
   BYTE paddingSize = 3 * (DimensionSqType)width % 4;
   BYTE padding = 0;
   DimensionSqType c;

   for(DimensionType y = 0; y < height; ++y)
   {
      c = y * width * 3;
      fwrite(&image[c], 1, width * 3, file);

      for(BYTE p = 0; p < paddingSize; ++p)
      {
         fwrite(&padding, 1, 1, file);
      }
   }

   fclose(file);
}

void saveAsBmp(BYTE* image, DimensionType width, DimensionType height,
               const char* filename)
{
   DimensionSqType resolution = width * height;

   BmpFileHeader fileHeader;
   fileHeader.filesize = 54 + (DWORD)resolution * 3
                         + (3 * (DWORD)width) % 4
                         * (DWORD)height;

   BmpInfoHeader infoHeader;
   infoHeader.width = (SDWORD)width;
   infoHeader.height = (SDWORD)height;
   infoHeader.imageSize = (DWORD)resolution * 3;

   FILE* file;
   file = fopen(filename, "wb");

   BYTE* buffer;
   buffer = (BYTE*) malloc(40);

   fileHeader.ToBytes(buffer);
   fwrite(buffer, 1, 14, file);

   infoHeader.ToBytes(buffer);
   fwrite(buffer, 1, 40, file);

   free(buffer);

   // Padding
   BYTE paddingSize = 3 * (DimensionSqType)width % 4;
   BYTE padding = 0;
   DimensionSqType c;

   for(DimensionType y = 0; y < height; y++)
   {
      c = y * width * 3;

      fwrite(&image[c], 1, width * 3, file);

      for(BYTE p = 0; p < paddingSize; p++)
      {
         fwrite(&padding, 1, 1, file);
      }
   }

   fclose(file);
}