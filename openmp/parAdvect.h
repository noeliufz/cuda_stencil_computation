// OpenMP 2D advection solver module

// sets up parallel parameters above
void init_parallel_parameter_values(int M, int N, int P, int Q, int verbosity);

// evolve advection over r timesteps, with (u,ldu) storing the local field
// using a 1D decomposition
void run_parallel_omp_advection_1D_decomposition(int r, double *u, int ldu);

// 2D, wide parallel region variant
void run_parallel_omp_advection_2D_decomposition(int r, double *u, int ldu);

// extra optimization variant
void run_parallel_omp_advection_with_extra_opts(int r, double *u, int ldu);
