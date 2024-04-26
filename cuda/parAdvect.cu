// CUDA parallel 2D advection solver module

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

//sets up parameters above
void init_parallel_parameter_values(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_, 
		   int verb) {
  M = M_, N = N_; Gx = Gx_; Gy = Gy_;  Bx = Bx_; By = By_; 
  verbosity = verb;
} //init_parallel_parameter_values()


__host__ __device__
static void calculate_and_update_coefficients(double v, double *cm1, double *c0, double *cp1) {
  double v2 = v/2.0;
  *cm1 = v2*(v+1.0);
  *c0  = 1.0 - v*v;
  *cp1 = v2*(v-1.0);
}


// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void run_parallel_cuda_advection_2D_decomposition(int reps, double *u, int ldu) {

} //run_parallel_cuda_advection_2D_decomposition()



// ... optimized parallel variant
void run_parallel_cuda_advection_optimized(int reps, double *u, int ldu, int w) {

} //run_parallel_cuda_advection_optimized()
