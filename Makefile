CC=g++
CFLAGS=-c -Wall

all: fractal

fractal: main.o fractal.o bmp.o common.o
	$(CC) main.o fractal.o bmp.o common.o -o fractal

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

fractal.o: fractal.cpp
	$(CC) $(CFLAGS) fractal.cpp

bmp.o: bmp.cpp
	$(CC) $(CFLAGS) bmp.cpp
	
common.o: common.cpp
	$(CC) $(CFLAGS) common.cpp
   
clean:
	rm -rf *o fractal
