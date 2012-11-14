#ifndef BMP_H
#define BMP_H

#include "common.h"

class BmpFileHeader {
public:
   WORD  identifier;    // 0x4D42 ("BM") for Windows
   DWORD filesize;
   WORD  reserved1;
   WORD  reserved2;
   DWORD offset;        // 54 bytes

   BmpFileHeader();
   void ToBytes(BYTE* buffer);
};

class BmpInfoHeader {
public:
   DWORD   infoHeaderSize;      // 40 bytes
   SDWORD    width;
   SDWORD    height;
   WORD   numColourPlanes;     // Always 1
   WORD   bitsPerPixel;        // Colour depth (normally 8)
   DWORD   compressionMethod;   // 0 for none
   DWORD   imageSize;           // 0 or filesize - 54
   DWORD    widthPixelsPerMeter; // 2835 is 72 dpi
   DWORD    heightPixelsPerMeter;// 2835 is 72 dpi
   DWORD   numColours;          // Number of colours in header palette (normally 0 for no palette)
   DWORD   numImportantColours; // usually 0 (ignored)

   BmpInfoHeader();
   void ToBytes(BYTE* buffer);
};

void saveAsBmp(BYTE* image, DimensionType width, DimensionType height,
               const char* filename);

void startBmp(DimensionType width, DimensionType height,
               const char* filename);
void appendBmp(BYTE* image, DimensionType width, DimensionType height,
               const char* filename);

#endif
