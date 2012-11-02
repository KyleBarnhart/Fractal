#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "bmp.h"

// BMP code thanks to deusmacabre
// http://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries
void saveAsBmp(const unsigned char* img, int w, int h,
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