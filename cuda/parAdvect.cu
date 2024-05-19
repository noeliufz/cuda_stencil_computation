// CUDA parallel 2D advection solver module

#include "serAdvect.h" // advection parameters
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

// sets up parameters above
void init_parallel_parameter_values(int M_, int N_, int Gx_, int Gy_, int Bx_,
                                    int By_, int verb) {
  M = M_, N = N_;
  Gx = Gx_;
  Gy = Gy_;
  Bx = Bx_;
  By = By_;
  verbosity = verb;
} // init_parallel_parameter_values()

__host__ __device__ static void calculate_and_update_coefficients(double v,
                                                                  double *cm1,
                                                                  double *c0,
                                                                  double *cp1) {
  double v2 = v / 2.0;
  *cm1 = v2 * (v + 1.0);
  *c0 = 1.0 - v * v;
  *cp1 = v2 * (v - 1.0);
}

/********************* Simple approach ******************************/

__global__ void par_update_north_south_boundary(int M, int N, double *u,
                                                int ldu) {
  int thread_dim_x = blockDim.x * gridDim.x;
  int thread_dim_y = blockDim.y * gridDim.y;
  int global_thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int global_thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
  int global_thread_id = global_thread_id_y * thread_dim_x +
                         global_thread_id_x; // thread global ID in 1D

  int n = N / (thread_dim_x * thread_dim_y);

  int x_start = global_thread_id * n;
  int x_end = global_thread_id < thread_dim_x * thread_dim_y - 1
                  ? (global_thread_id + 1) * n
                  : N + 2;
  // printf("n: %d, x start: %d, x end: %d thread: %d\n", n, x_start, x_end,
  //      global_thread_id);

  for (int i = x_start; i < x_end; i++) {
    if (i == 0 || i == N + 1)
      ;
    V(u, 0, i) = V(u, M, i);
    V(u, M + 1, i) = V(u, 1, i);
    // printf("Updated column %d by thread %d\n", i + 1,
    // global_thread_id);
  }
}

__global__ void par_update_east_west_boundary(int M, int N, double *u,
                                              int ldu) {

  int thread_dim_x = blockDim.x * gridDim.x;
  int thread_dim_y = blockDim.y * gridDim.y;
  int global_thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int global_thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
  int global_thread_id = global_thread_id_y * thread_dim_x +
                         global_thread_id_x; // thread global ID in 1D

  int n = M / (thread_dim_x * thread_dim_y);

  int y_start = global_thread_id * n;
  int y_end = global_thread_id < thread_dim_x * thread_dim_y - 1
                  ? (global_thread_id + 1) * n
                  : M + 2;
  // printf("n: %d, y start: %d, y end: %d thread: %d\n", n, y_start, y_end,
  //      global_thread_id);

  for (int i = y_start; i < y_end; i++) {
    if (i == 0 || i == M + 1)
      ;
    V(u, i, 0) = V(u, i, N);
    V(u, i, N + 1) = V(u, i, 1);
    // printf("Updated column %d by thread %d\n", i + 1,
    // global_thread_id);
  }
}

__global__ void par_update_advection_field_kernel(int M, int N, double *u,
                                                  int ldu, double *v, int ldv,
                                                  double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);
  // Global thread value, x and y
  int thread_dim_x = blockDim.x * gridDim.x;
  int thread_dim_y = blockDim.y * gridDim.y;
  // printf("thread dim x: %d, thread dim y: %d\n", thread_dim_x,
  // thread_dim_y);
  int global_thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int global_thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
  int global_thread_id = global_thread_id_y * thread_dim_x +
                         global_thread_id_x; // thread global ID in 1D
  // Divide the matrix into (n_x * n_y) parts
  int n_x = M / thread_dim_x;
  int n_y = N / thread_dim_y;
  // Let all threads do its update, specify with its own start index and end
  // index
  //
  int x_start = global_thread_id_x * n_x;
  int x_end = global_thread_id_x < thread_dim_x - 1
                  ? (global_thread_id_x + 1) * n_x
                  : M;
  int y_start = global_thread_id_y * n_y;
  int y_end = global_thread_id_y < thread_dim_y - 1
                  ? (global_thread_id_y + 1) * n_y
                  : N;

  // printf("Thread %d, x from %d to %d, y from %d to %d\n", thread_id, x_start,
  //     x_end, y_start, y_end);

  for (int i = x_start; i < x_end; i++)
    for (int j = y_start; j < y_end; j++) {
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
      // printf("accessing i: %d, j: %d\n", i, j);
    }
}

__global__ void par_copy_field_kernel(int M, int N, double *v, int ldv,
                                      double *u, int ldu) {

  // Global thread value, x and y
  int thread_dim_x = blockDim.x * gridDim.x;
  int thread_dim_y = blockDim.y * gridDim.y;
  // printf("thread dim x: %d, thread dim y: %d\n", thread_dim_x,
  // thread_dim_y);
  int global_thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int global_thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
  int global_thread_id = global_thread_id_y * thread_dim_x +
                         global_thread_id_x; // thread global ID in 1D
  // Divide the matrix into (n_x * n_y) parts
  int n_x = M / thread_dim_x;
  int n_y = N / thread_dim_y;
  // Let all threads do its update, specify with its own start index and end
  // index
  //
  int x_start = global_thread_id_x * n_x;
  int x_end = global_thread_id_x < thread_dim_x - 1
                  ? (global_thread_id_x + 1) * n_x
                  : M;
  int y_start = global_thread_id_y * n_y;
  int y_end = global_thread_id_y < thread_dim_y - 1
                  ? (global_thread_id_y + 1) * n_y
                  : N;
  // printf("Updating x from %d to %d\n", x_start, x_end);
  for (int i = x_start; i < x_end; i++)
    for (int j = y_start; j < y_end; j++) {
      u[i * ldu + j] = v[i * ldv + j];
      // printf("Updating %d to %d\n", i * ldu + j, i * ldv + j);
    }
}

/********************* Optimized approach ******************************/
// Map shared memory index to global memory index (with halo)
__device__ void index_shared_to_global(int M, int N, int i, int j, int *gi,
                                       int *gj) {
  i--;
  j--;
  *gi = blockIdx.x * M / gridDim.x + i;
  *gj = blockIdx.y * N / gridDim.y + j;
  *gi = *gi + 1;
  *gj = *gj + 1;
  i++;
  j++;
}
__device__ void fix_index_boundary(int M, int N, int *i, int *j) {
  int gi, gj;
  index_shared_to_global(M, N, *i, *j, &gi, &gj);
  if (gi >= M + 2) {
    *i = *i - (gi - M - 2);
  }
  if (gj >= N + 2) {
    *j = *j - (gj - N - 2);
  }
}
__device__ void init_shared_memory(double *shared_u, int ldbu, int M, int N,
                                   double *u, int ldu) {
  int total_i_start = 0;
  int total_i_end = total_i_start + M / gridDim.y / blockDim.y + 1;
  int total_j_start = 0;
  int total_j_end = total_j_start + N / gridDim.x / blockDim.x + 1;

  int size_i = total_i_end / blockDim.y;
  int size_j = total_j_end / blockDim.x;
  int shared_i_start = threadIdx.x * size_i;
  int shared_j_start = threadIdx.y * size_j;
  int shared_i_end = (threadIdx.x + 1) * size_i;
  int shared_j_end = (threadIdx.y + 1) * size_j;

  fix_index_boundary(M, N, &shared_i_end, &shared_j_end);
  // printf("i %d,%d j %d,%d\n", shared_i_start, shared_j_start, shared_i_start,
  //      shared_i_end);

  for (int i = shared_i_start; i < shared_i_end; i++) {
    for (int j = shared_j_start; j < shared_j_end; j++) {
      int gi, gj;
      index_shared_to_global(M, N, i, j, &gi, &gj);
      shared_u[ldbu * i + j] = u[ldu * gi + gj];
    }
  }
}
__device__ void opt_update_north_south_boundary(double *shared_u,
                                                double *shared_v, int ldbu,
                                                int M, int N, double *u,
                                                int ldu) {
  int total_j_start = 0;
  int total_j_end = total_j_start + N / gridDim.x / blockDim.x + 1;

  int size = total_j_end / blockDim.x / blockDim.y;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.y + 1;
  int shared_j_start = thread_id * size;
  int shared_j_end = (thread_id + 1) * size;
  int shared_i_end = M / blockDim.y / gridDim.y;

  fix_index_boundary(M, N, &shared_i_end, &shared_j_end);
  for (int j = shared_j_start; j < shared_j_end; j++) {
    int gi, gj;
    int i = 0;
    index_shared_to_global(M, N, i, j + 1, &gi, &gj);
    u[ldu + gj] = shared_u[ldbu + j];
    __syncthreads();
    index_shared_to_global(M, N, i, j + 1, &gi, &gj);
    shared_u[j] = u[gj];
    __syncthreads();
    index_shared_to_global(M, N, shared_i_end, j + 1, &gi, &gj);
    u[ldu + gj + gi] = shared_u[ldbu + j + i];
    __syncthreads();
    index_shared_to_global(M, N, shared_i_end, j + 1, &gi, &gj);
    shared_u[ldbu * i + j] = u[ldu * gi + gj];
    __syncthreads();
  }
}
__device__ void opt_update_east_west_boundary(double *shared_u,
                                              double *shared_v, int ldbu, int M,
                                              int N, double *u, int ldu) {
  int total_i_start = 0;
  int total_i_end = total_i_start + M / gridDim.y / blockDim.y;

  int size = total_i_end / blockDim.x / blockDim.y;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.y;
  int shared_i_start = thread_id * size;
  int shared_i_end = (thread_id + 1) * size;
  int shared_j_end = N / blockDim.x / gridDim.x;

  fix_index_boundary(M, N, &shared_i_end, &shared_j_end);
  for (int i = shared_i_start; i < shared_i_end; i++) {
    int gi, gj;
    int j = 0;
    index_shared_to_global(M, N, i + 1, j + 1, &gi, &gj);
    u[(gi + 1) * ldu] = shared_v[(i + 1) * ldbu + 1];
    __syncthreads();
    shared_u[(i + 1) * ldbu] = u[(gi + 1) * ldu + 1];
    index_shared_to_global(M, N, shared_i_end, j, &gi, &gj);
    u[(gi + 1) * ldu + gj + 1] = shared_v[(i + 1) * ldbu + shared_j_end];
    __syncthreads();
    shared_u[(i + 1) * ldbu + shared_j_end + 1] = u[(gi + 1) * ldu + gj];
  }
}
__device__ void opt_update_advection_field_kernel(int M, int N, double *u,
                                                  int ldu, double *v, int ldv,
                                                  double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);

  int total_i_start = 0;
  int total_i_end = total_i_start + M / gridDim.y / blockDim.y;
  int total_j_start = 0;
  int total_j_end = total_j_start + N / gridDim.x / blockDim.x;

  int size_i = total_i_end / blockDim.y;
  int size_j = total_j_end / blockDim.x;
  int shared_i_start = threadIdx.x * size_i;
  int shared_j_start = threadIdx.y * size_j;
  int shared_i_end = (threadIdx.x + 1) * size_i;
  int shared_j_end = (threadIdx.y + 1) * size_j;

  for (int i = shared_i_start; i < shared_i_end; i++)
    for (int j = shared_j_start; j < shared_j_end; j++) {
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
}

__device__ void opt_copy_field_kernel(int M, int N, double *v, int ldv,
                                      double *u, int ldu) {

  int total_i_start = 0;
  int total_i_end = total_i_start + M / gridDim.y / blockDim.y + 1;
  int total_j_start = 0;
  int total_j_end = total_j_start + N / gridDim.x / blockDim.x + 1;

  int size_i = total_i_end / blockDim.y;
  int size_j = total_j_end / blockDim.x;
  int shared_i_start = threadIdx.x * size_i;
  int shared_j_start = threadIdx.y * size_j;
  int shared_i_end = (threadIdx.x + 1) * size_i;
  int shared_j_end = (threadIdx.y + 1) * size_j;
  for (int i = shared_i_start; i < shared_i_end; i++)
    for (int j = shared_j_start; j < shared_j_end; j++) {
      u[i * ldu + j] = v[i * ldv + j];
    }
}
__device__ void shared_memory_copy_back(double *shared_u, int ldbu, int M,
                                        int N, double *u, int ldu) {
  int total_i_start = 0;
  int total_i_end = total_i_start + M / gridDim.y / blockDim.y + 1;
  int total_j_start = 0;
  int total_j_end = total_j_start + N / gridDim.x / blockDim.x + 1;

  int size_i = total_i_end / blockDim.y;
  int size_j = total_j_end / blockDim.x;
  int shared_i_start = threadIdx.x * size_i;
  int shared_j_start = threadIdx.y * size_j;
  int shared_i_end = (threadIdx.x + 1) * size_i;
  int shared_j_end = (threadIdx.y + 1) * size_j;

  fix_index_boundary(M, N, &shared_i_end, &shared_j_end);
  // printf("i %d,%d j %d,%d\n", shared_i_start, shared_j_start, shared_i_start,
  //      shared_i_end);

  for (int i = shared_i_start; i < shared_i_end; i++) {
    for (int j = shared_j_start; j < shared_j_end; j++) {
      int gi, gj;
      index_shared_to_global(M, N, i, j, &gi, &gj);
      shared_u[ldbu * i + j] = u[ldu * gi + gj];
    }
  }
}

__global__ void run_opt(int M, int N, double *device_u, int ldu, double *v,

                        int ldv, double Ux, double Uy, int reps) {

  int ldbu = N / gridDim.x / blockDim.x + 2;
  // printf("ldbu: %d\n", ldbu);
  extern __shared__ double shared_u[];
  extern __shared__ double shared_v[];

  init_shared_memory(shared_u, ldbu, M, N, device_u, ldu);
  __syncthreads();

  for (int r = 0; r < reps; r++) {
    opt_update_north_south_boundary(shared_u, shared_v, ldbu, M, N, device_u,
                                    ldu);
    opt_update_east_west_boundary(shared_u, shared_v, ldbu, M, N, device_u,
                                  ldu);
    opt_update_advection_field_kernel(M, N, &shared_u[ldbu + 1], ldbu,
                                      &shared_v[ldbu + 1], ldbu, Ux, Uy);
    opt_copy_field_kernel(M, N, &shared_v[ldbu + 1], ldbu, &shared_u[ldbu + 1],
                          ldbu);
    __syncthreads();
  } // for(r...)

  shared_memory_copy_back(shared_u, ldbu, M, N, device_u, ldu);
}

// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void run_parallel_cuda_advection_2D_decomposition(int reps, double *u,
                                                  int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N + 2;
  double *v, *device_u;
  HANDLE_ERROR(cudaMalloc(&v, ldv * (M + 2) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&device_u, ldv * (M + 2) * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(device_u, u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyHostToDevice));

  dim3 grid(Gx, Gy);
  dim3 block(Bx, By);
  // dim3 grid(1, 1);
  // dim3 block(1, 1);
  for (int r = 0; r < reps; r++) {
    par_update_north_south_boundary<<<grid, block>>>(M, N, device_u, ldu);
    par_update_east_west_boundary<<<grid, block>>>(M, N, device_u, ldu);
    par_update_advection_field_kernel<<<grid, block>>>(
        M, N, &device_u[ldu + 1], ldu, &v[ldv + 1], ldv, Ux, Uy);
    par_copy_field_kernel<<<grid, block>>>(M, N, &v[ldv + 1], ldv,
                                           &device_u[ldu + 1], ldu);
  } // for(r...)
  HANDLE_ERROR(cudaMemcpy(u, device_u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(v));
  HANDLE_ERROR(cudaFree(device_u));
} // run_parallel_cuda_advection_2D_decomposition()

// ... optimized parallel variant
void run_parallel_cuda_advection_optimized(int reps, double *u, int ldu,
                                           int w) {

  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N + 2;
  double *v, *device_u;
  HANDLE_ERROR(cudaMalloc(&v, ldv * (M + 2) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&device_u, ldv * (M + 2) * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(device_u, u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyHostToDevice));
  dim3 grid(Gx, Gy);
  dim3 block(Bx, By);

  int size = (M / Gy / By + 2) * (N / Gx / Bx + 2) * sizeof(double);
  printf("size: %d\n", size);
  run_opt<<<grid, block, size>>>(M, N, device_u, ldu, v, ldv, Ux, Uy, reps);

  HANDLE_ERROR(cudaMemcpy(u, device_u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(v));
  HANDLE_ERROR(cudaFree(device_u));
} // run_parallel_cuda_advection_optimized()
