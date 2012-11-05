=======
Fractal
=======

Fractal Image Program

Usage
------
* Resolution
    * `-res <width> <height>`
	* Each must be 4 or more
	* Default: 800 600
	* Example: `-res 1920 1080`

* Iterations
    * `-i <number of iterations>`
	* Must be integer of 1 or more
	* Use more iterations the deeper you zoom
	* Default: 50
	* Example: `-i 100`
	
* Position
    * `-x <x coordinate> -y <y coordinate>`
	* Default: -0.5 0
	* Example: `-x 0.001643721969 -y -0.8224676332991`

* Zoom
    * `-x <zoom factor>`
	* Working range is 1 to 10^15
	* Default: 1
	* Example: `-z 2.5`

* Anti-Aliasing
    * `-ssaa <factor>`
    * Selective super sampling
	* Detects edges
	* Must be an integer or 2 or more
	* Typically 2 to 4
	* Default: 0
	* Example: `-ssaa 2`

* Sequence of Images
    * `-sequence <end zoom factor> <frames>`
	* Results in a number of images equal to frames
	* Zooms from zoom factor `-z` to end zoom factor
	* Example: `-sequence 100000000000 720`

### Usage Examples
`./fractal -res 1920 1080 -i 1500 -ssaa 3 -x 0.001643721969 -y -0.8224676332991 -z 100000000000000`
`./fractal -res 1920 1080 -i 1500 -ssaa 3 -x 0.001643721969 -y -0.8224676332991 -sequence 100000000000000 720`
