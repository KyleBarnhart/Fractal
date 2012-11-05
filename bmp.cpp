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
   infoHeader.imageSize = resolution * 3;

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

      for(DimensionSqType x = 0; x < width ; x++)
      {
         fwrite(&image[c + x*3 + 2], 1, 1, file);//Blue
         fwrite(&image[c + x*3 + 1], 1, 1, file); //Green
         fwrite(&image[c + x*3], 1, 1, file); //Red
      }
      for(BYTE p = 0; p < paddingSize; p++)
      {
         fwrite(&padding, 1, 1, file);
      }
   }

   fclose(file);
}

// BMP code thanks to deusmacabre
// http://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries
void saveAsBmpOld(const unsigned char* img, int w, int h,
               const char* filename)
{
   FILE* f;
   int filesize = 54 + 3*w*h;  //w is your image width, h is image height, both int

   unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
   unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
   unsigned char bmppad[3] = {0,0,0};

   bmpfileheader[ 2] = (unsigned char)(filesize    );
   bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
   bmpfileheader[ 4] = (unsigned char)(filesize>>16);
   bmpfileheader[ 5] = (unsigned char)(filesize>>24);

   bmpinfoheader[ 4] = (unsigned char)(       w    );
   bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
   bmpinfoheader[ 6] = (unsigned char)(       w>>16);
   bmpinfoheader[ 7] = (unsigned char)(       w>>24);
   bmpinfoheader[ 8] = (unsigned char)(       h    );
   bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
   bmpinfoheader[10] = (unsigned char)(       h>>16);
   bmpinfoheader[11] = (unsigned char)(       h>>24);

   f = fopen(filename, "w");
   fwrite(bmpfileheader,1,14,f);
   fwrite(bmpinfoheader,1,40,f);
   for(int i=0; i<h; i++)
   {
       fwrite(img+(w*(h-i-1)*3),3,w,f);
       fwrite(bmppad,1,(4-(w*3)%4)%4,f);
   }
   fclose(f);
}