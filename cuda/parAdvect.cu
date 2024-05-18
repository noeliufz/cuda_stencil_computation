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

/********************* Naive approach ******************************/
__global__ void par_update_north_south_boundary(int M, int N, double *u,
                                                int ldu) {
  int thread_dim_x = blockDim.x * gridDim.x;
  int thread_dim_y = blockDim.y * gridDim.y;
  int thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
  int thread_id =
      thread_id_y * thread_dim_x + thread_id_x; // thread global ID in 1D
  // printf("Global ID: %d\n", idx);

  int i = thread_id;
  // Let all the threads update N elements
  // Use i range from 0 to N for loop, but i+1 to visit element to avoid
  // update 0 and N+1.
  while (i < N + 2) {
    if (i == 0 || i == N + 1)
      ;
    V(u, 0, i) = V(u, M, i);
    V(u, M + 1, i) = V(u, 1, i);
    // printf("Updated column %d by thread %d\n", i + 1, idx);
    i = i + thread_dim_x * thread_dim_y;
  }
}

__global__ void par_update_east_west_boundary(int M, int N, double *u,
                                              int ldu) {
  int thread_dim_x = blockDim.x * gridDim.x;
  int thread_dim_y = blockDim.y * gridDim.y;
  int thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
  int thread_id =
      thread_id_y * thread_dim_x + thread_id_x; // thread global ID in 1D

  int i = thread_id;
  while (i < M + 2) {
    if (i == 0 || i == M + 1)
      ;
    V(u, i, 0) = V(u, i, N);
    V(u, i, N + 1) = V(u, i, 1);
    // printf("Updated line %d by thread %d\n", i, thread_id);
    i = i + thread_dim_x * thread_dim_y;
  }
}

__global__ void par_update_advection_field_kernel(int M, int N, double *u,
                                                  int ldu, double *v, int ldv,
                                                  double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);
  // Global thread x and y
  int thread_dim_x = blockDim.x * gridDim.x;
  int thread_dim_y = blockDim.y * gridDim.y;
  // printf("thread dim x: %d, thread dim y: %d\n", thread_dim_x,
  // thread_dim_y);
  int thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
  int thread_id =
      thread_id_y * thread_dim_x + thread_id_x; // thread global ID in 1D
  // Divide the matrix into (n_x * n_y) parts
  int n_x = M / thread_dim_x;
  int n_y = N / thread_dim_y;
  // Let all threads do its update, specify with its own start index and end
  // index
  //
  int x_start = thread_id_x * n_x;
  int x_end = thread_id_x < thread_dim_x - 1 ? (thread_id_x + 1) * n_x : M;
  int y_start = thread_id_y * n_y;
  int y_end = thread_id_y < thread_dim_y - 1 ? (thread_id_y + 1) * n_y : N;

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
  // Global thread x and y
  int thread_dim_x = blockDim.x * gridDim.x;
  int thread_dim_y = blockDim.y * gridDim.y;
  int thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
  // Divide the matrix into (n_x * n_y) parts
  int n_x = M / thread_dim_x;
  int n_y = N / thread_dim_y;
  // Let all threads do its update, specify with its own start index and end
  // index
  int x_start = thread_dim_x * n_x;
  int x_end = thread_id_x < n_x - 1 ? (thread_id_x + 1) * n_x : M;
  int y_start = thread_dim_y * n_y;
  int y_end = thread_id_y < n_y - 1 ? (thread_id_y + 1) * n_y : N;
  for (int i = x_start; i < x_end; i++)
    for (int j = y_start; j < y_end; j++)
      u[i * ldu + j] = v[i * ldv + j];
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
    copy_field_kernel<<<grid, block>>>(M, N, &v[ldv + 1], ldv,
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

} // run_parallel_cuda_advection_optimized()
