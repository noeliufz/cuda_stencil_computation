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

  int i_start = global_thread_id * n;
  int i_end = global_thread_id < thread_dim_x * thread_dim_y - 1
                  ? (global_thread_id + 1) * n
                  : N + 2;

  for (int i = i_start; i < i_end; i++) {
    if (i == 0 || i == N + 1)
      continue;
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

  int j_start = global_thread_id * n;
  int j_end = global_thread_id < thread_dim_x * thread_dim_y - 1
                  ? (global_thread_id + 1) * n
                  : M + 2;
  // printf("n: %d, y start: %d, y end: %d thread: %d\n", n, y_start, y_end,
  //      global_thread_id);

  for (int j = j_start; j < j_end; j++) {
    if (j == 0 || j == M + 1)
      continue;
    V(u, j, 0) = V(u, j, N);
    V(u, j, N + 1) = V(u, j, 1);
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

  int global_thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int global_thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;
  int global_thread_id = global_thread_id_y * thread_dim_x +
                         global_thread_id_x; // thread global ID in 1D
  // Divide the matrix into (n_x * n_y) parts
  int n_i = M / thread_dim_x;
  int n_j = N / thread_dim_y;
  // Let all threads do its update, specify with its own start index and end
  // index
  int i_start = global_thread_id_x * n_i;
  int i_end = global_thread_id_x < thread_dim_x - 1
                  ? (global_thread_id_x + 1) * n_i
                  : M;
  int j_start = global_thread_id_y * n_j;
  int j_end = global_thread_id_y < thread_dim_y - 1
                  ? (global_thread_id_y + 1) * n_j
                  : N;

  for (int i = i_start; i < i_end; i++)
    for (int j = j_start; j < j_end; j++) {
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
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
  int n_i = M / thread_dim_x;
  int n_j = N / thread_dim_y;
  // Let all threads do its update, specify with its own start index and end
  // index
  //
  int i_start = global_thread_id_x * n_i;
  int i_end = global_thread_id_x < thread_dim_x - 1
                  ? (global_thread_id_x + 1) * n_i
                  : M;
  int j_start = global_thread_id_y * n_j;
  int j_end = global_thread_id_y < thread_dim_y - 1
                  ? (global_thread_id_y + 1) * n_j
                  : N;
  // printf("Updating x from %d to %d\n", x_start, x_end);
  for (int i = i_start; i < i_end; i++)
    for (int j = j_start; j < j_end; j++) {
      u[i * ldu + j] = v[i * ldv + j];
      // printf("Updating %d to %d\n", i * ldu + j, i * ldv + j);
    }
}

/********************* Optimized approach ******************************/
__global__ void opt_update_advection_field_kernel(int M, int N, double *u,
                                                  int ldu, double *v, int ldv,
                                                  double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);

  extern __shared__ double shared_u[];
  int ldshared_u = blockDim.y + 2;

  int x = threadIdx.x;
  int y = threadIdx.y;
  int global_thread_id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int global_thread_id_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (global_thread_id_x < M + 1 && global_thread_id_y < N + 1) {
    // copy from global memory to shared memory for threads within one block
    V(shared_u, threadIdx.x + 1, threadIdx.y + 1) =
        V(u, global_thread_id_x + 1, global_thread_id_y + 1);

    // update shared memory halo

    if (threadIdx.x == 0) {
      V(shared_u, 0, threadIdx.y + 1) =
          V(u, global_thread_id_x, global_thread_id_y + 1);
      if (threadIdx.y == 0) {
        V(shared_u, 0, 0) = V(u, global_thread_id_x, global_thread_id_y);
      }
      if (threadIdx.y == blockDim.y - 1 || global_thread_id_y == N - 1) {
        V(shared_u, 0, threadIdx.y + 2) =
            V(u, global_thread_id_x, global_thread_id_y + 2);
      }
    }
    if (threadIdx.x == blockDim.x - 1 || global_thread_id_x == M - 1) {
      V(shared_u, threadIdx.x + 2, threadIdx.y + 1) =
          V(u, global_thread_id_x + 2, global_thread_id_y + 1);
      if (threadIdx.y == 0) {
        V(shared_u, threadIdx.x + 2, 0) =
            V(u, global_thread_id_x + 2, global_thread_id_y);
      }
      if (threadIdx.y == blockDim.y - 1 || global_thread_id_y == N - 1) {
        V(shared_u, threadIdx.x + 2, threadIdx.y + 2) =
            V(u, global_thread_id_x + 2, global_thread_id_y + 2);
      }
    }

    // west and east
    if (threadIdx.y == 0) {
      V(shared_u, threadIdx.x + 1, 0) =
          V(u, global_thread_id_x + 1, global_thread_id_y);
    }
    if (threadIdx.y == blockDim.y - 1 || global_thread_id_y == N - 1) {
      V(shared_u, threadIdx.x + 1, threadIdx.y + 2) =
          V(u, global_thread_id_x + 1, global_thread_id_y + 2);
    }
    __syncthreads();

    V(v, global_thread_id_x + 1, global_thread_id_y + 1) =
        cim1 * (cjm1 * V(shared_u, threadIdx.x, threadIdx.y) +
                cj0 * V(shared_u, threadIdx.x, threadIdx.y + 1) +
                cjp1 * V(shared_u, threadIdx.x, threadIdx.y + 2)) +
        ci0 * (cjm1 * V(shared_u, threadIdx.x + 1, threadIdx.y) +
               cj0 * V(shared_u, threadIdx.x + 1, threadIdx.y + 1) +
               cjp1 * V(shared_u, threadIdx.x + 1, threadIdx.y + 2)) +
        cip1 * (cjm1 * V(shared_u, threadIdx.x + 2, threadIdx.y) +
                cj0 * V(shared_u, threadIdx.x + 2, threadIdx.y + 1) +
                cjp1 * V(shared_u, threadIdx.x + 2, threadIdx.y + 2));
  }
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

  // round up to make sure cover all data
  int gridDimX = (M + Bx - 1) / Bx;
  int gridDimY = (N + By - 1) / By;

  int size = (M / Gy / By + 2) * (N / Gx / Bx + 2) * sizeof(double);

  dim3 grid(gridDimX, gridDimY);
  dim3 block(Bx, By);

  for (int r = 0; r < reps; r++) {
    par_update_north_south_boundary<<<grid, block>>>(M, N, device_u, ldu);
    par_update_east_west_boundary<<<grid, block>>>(M, N, device_u, ldu);
    opt_update_advection_field_kernel<<<grid, block, size>>>(
        M, N, device_u, ldu, v, ldv, Ux, Uy);
    cudaDeviceSynchronize();

    // swap buffer pointer
    double *temp = device_u;
    device_u = v;
    v = temp;
  }

  // if there is only odd times of computations, swap again to make sure the u
  // is pointing to the updated data
  if (reps % 2 == 1) {
    double *temp = device_u;
    device_u = v;
    v = temp;
  }

  HANDLE_ERROR(cudaMemcpy(u, device_u, ldv * (M + 2) * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(v));
  HANDLE_ERROR(cudaFree(device_u));
} // run_parallel_cuda_advection_optimized()
