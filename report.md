# Cuda
## Naïve approach

### Update boundary
I turned the threads to 1D fashion to divid the `N` elements and `M+2` elements to all threads to update. There are `n=blockDim.x*blockDim.y*gridDim.x*gridDim.y` threads so each thread will update `2*N\n` data for both top and bottom and `2*(M+2)\n` elements for both left and right boundary.


1,1,512,1
1,1,64,8
1,1,16,16
1,1,8,64
1,1,1,512
4,4,512,1
4,4,64,8
4,4,16,16
4,4,8,64
4,4,1,512
8,8,512,1
8,8,64,8
8,8,16,16
8,8,8,64
8,8,1,512
16,16,512,1
16,16,64,8
16,16,16,16
16,16,8,64
16,16,1,512








## Optimized approach
### Pointer swap
According to Vizitiu et al., the pointer to original data and caculated data (in this project array `u` and array `v) can be swapped to save copy time.

### Shared memory within one block
In naïve approach, all threads read data from global memory of the GPU which spends more time than reading from a shared memory. 

As data in shared memory are shared with all threads within one block, I created shared memory for each block and copy the data from global memory to shared memory before computing and loaded extra rows and columns as halo. After computation, each thread save computed data back to global memory `v`. After each step, using pointer swap indicated in previous paragraph, swap pointer of `u` and `v` to save copy time.