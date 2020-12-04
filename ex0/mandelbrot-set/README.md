# Mandelbrot set
The Mandelbrot set is the set of complex numbers $`c`$ for which the function $`f_{c}(z)=z^{2}+c`$ $`f_{c}(z)=z^{2}+c`$ does not diverge when iterated from $`z=0`$, i.e., for which the sequence $`f_{c}(0)`$ $`f_{c}(f_{c}(0))`$, etc., remains bounded in absolute value.


## Requirements

Make sure that you have the following packages installed on yout system:
```
gcc >= 5.5.0
cmake >= 3.5
cuda 8.0
boost 1.65.1
libjpeg-turbo 2.0.4
```

## How to compile
```
cd <mandelbrot set root>
mkdir build && cd build
cmake ..
make
```

##### Additional compile options
| Name 	        | Values        	|
|---	        |---  	            |
| PRECISION 	| single, double	|
| SM 	        | sm_20, sm_60      |

Example:
```
cmake -DPRECISION=single -DSM=sm_20 ..
```

## Program Options
```
./mset --help
Allowed options:
  -h [ --help ]                     produce help message
  -x [ --center_x ] arg (=-0.75)    domain center along x
  -y [ --center_y ] arg (=-0.25)    domain center along y
  -W [ --width ] arg (=3.5)         domain width
  -H [ --height ] arg (=4)          domain height
  --it arg (=1000)                  max. num. iterations
  --resX arg (=6400)                num. pixels along x direction
  --resY arg (=4800)                num. pixels along y direction
  -q [ --quality ] arg (=100)       image quality
  -o [ --output ] arg (=./set.jpeg) output file
  -m [ --mode ] arg (=serial)       supported compute mode: serial, threading, 
                                    gpu
```

## Running
#### Serial
```
./mset 
```

#### With threading
```
./mset -m threading
```

Note! It is highly recomended to set up the following environment variables before running the application:
```
export OMP_NUM_THREADS=<num avaliable cores>
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

#### With gpu
```
./mset -m gpu
```

If you have multiple **but different** GPUs installed on your system you can test both of them. Your GPUs are enumerated from $0$ to $n - 1$. By default, CUDA assins *id 0* to the "best" GPU. You need to set the following environment variable to force CUDA to pick up another GPU:
```
export CUDA_VISIBLE_DEVICES=<i>
```

You can inspect your installed GPU with the following command:
```
nvidia-smi
```

## What to test
Feel free to play with the all input parameters.

- Compute the speed-up
- Try to increase the maximum number of iterations
- Try to compile and run with both double and single precision
- Try to understand why elapsed and compute time are so different in case of GPU computing