// serial 2D advection solver module

#include <assert.h>
#include <math.h> // sin(), fabs()
#include <stdio.h>
#include <stdlib.h>

#include "serAdvect.h"

void cudaHandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

// advection parameters
const double CFL = 0.25;             // CFL condition number
const double Velx = 1.0, Vely = 1.0; // advection velocity
double dt;                           // time for 1 step
double deltax, deltay;               // grid spacing

void init_advection_parameters(int M, int N) {
  assert(M > 0 && N > 0); // advection not defined for empty grids
  deltax = 1.0 / N;
  deltay = 1.0 / M;
  dt = CFL * (deltax < deltay ? deltax : deltay);
}

double evaluate_initial_condition(double x, double y, double t) {
  x = x - Velx * t;
  y = y - Vely * t;
  return (sin(4.0 * M_PI * x) * sin(2.0 * M_PI * y));
}

void init_advection_field(int M, int N, double *u, int ldu) {
  int i, j;
  for (i = 0; i < M; i++) {
    double y = deltay * i;
    for (j = 0; j < N; j++) {
      double x = deltax * j;
      u[i * ldu + j] = evaluate_initial_condition(x, y, 0.0);
    }
  }
} // init_advection_field()

double compute_error_advection_field(int r, int M, int N, double *u, int ldu) {
  int i, j;
  double err = 0.0;
  double t = r * dt;
  for (i = 0; i < M; i++) {
    double y = deltay * i;
    for (j = 0; j < N; j++) {
      double x = deltax * j;
      err += fabs(u[i * ldu + j] - evaluate_initial_condition(x, y, t));
    }
  }
  return (err);
} // compute_error_advection_field()

double compute_max_error_advection_field(int r, int M, int N, double *u,
                                         int ldu) {
  int i, j;
  double err = 0.0;
  double t = r * dt;
  for (i = 0; i < M; i++) {
    double y = deltay * i;
    for (j = 0; j < N; j++) {
      double x = deltax * j;
      double e = fabs(u[i * ldu + j] - evaluate_initial_condition(x, y, t));
      if (e > err)
        err = e;
    }
  }
  return (err);
} // compute_max_error_advection_field()

void print_advection_field(std::string label, int M, int N, double *u,
                           int ldu) {
  int i, j;
  printf("%s\n", label.c_str());
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++)
      printf(" %+0.2f", u[i * ldu + j]);
    printf("\n");
  }
}

const int advection_flops_per_element = 20; // count 'em

__host__ __device__ void calculate_and_update_coefficients(double v,
                                                           double *cm1,
                                                           double *c0,
                                                           double *cp1) {
  double v2 = v / 2.0;
  *cm1 = v2 * (v + 1.0);
  *c0 = 1.0 - v * v;
  *cp1 = v2 * (v - 1.0);
}

__host__ __device__ void update_advection_field(int M, int N, double *u,
                                                int ldu, double *v, int ldv,
                                                double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);

} // update_advection_field()

__host__ __device__ void copy_field(int M, int N, double *v, int ldv, double *u,
                                    int ldu) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      u[i * ldu + j] = v[i * ldv + j];
}

void update_boundary(int M, int N, double *u, int ldu) {
  for (int j = 1; j < N + 1; j++) { // top and bottom halo
    u[j] = u[M * ldu + j];
    u[(M + 1) * ldu + j] = u[ldu + j];
  }
  for (int i = 0; i < M + 2; i++) { // left and right sides of halo
    u[i * ldu] = u[i * ldu + N];
    u[i * ldu + N + 1] = u[i * ldu + 1];
  }
}

void run_serial_advection_host(int M, int N, int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N + 2;
  double *v = (double *)calloc(ldv * (M + 2), sizeof(double));
  assert(v != NULL);
  for (int r = 0; r < reps; r++) {
    update_boundary(M, N, u, ldu);
    update_advection_field(M, N, &u[ldu + 1], ldu, &v[ldv + 1], ldv, Ux, Uy);
    copy_field(M, N, &v[ldv + 1], ldv, &u[ldu + 1], ldu);
  } // for(r...)
  free(v);
} // run_serial_advection_host()

/********************** serial GPU area **********************************/

__global__ void update_north_south_boundary(int N, int M, double *u, int ldu) {
  for (int j = 1; j < N + 1; j++) { // top and bottom halo
    u[j] = u[M * ldu + j];
    u[(M + 1) * ldu + j] = u[ldu + j];
  }
}

__global__ void update_east_west_boundary(int M, int N, double *u, int ldu) {
  for (int i = 0; i < M + 2; i++) { // left and right sides of halo
    u[i * ldu] = u[i * ldu + N];
    u[i * ldu + N + 1] = u[i * ldu + 1];
  }
}

__global__ void update_advection_field_kernel(int M, int N, double *u, int ldu,
                                              double *v, int ldv, double Ux,
                                              double Uy) {
  update_advection_field(M, N, u, ldu, v, ldv, Ux, Uy);
}

__global__ void copy_field_kernel(int M, int N, double *u, int ldu, double *v,
                                  int ldv) {
  copy_field(M, N, u, ldu, v, ldv);
}

// evolve advection over reps timesteps, with (u,ldu) containing the field
void run_serial_advection_device(int M, int N, int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N + 2;
  double *v;
  HANDLE_ERROR(cudaMalloc(&v, ldv * (M + 2) * sizeof(double)));
  for (int r = 0; r < reps; r++) {
    update_north_south_boundary<<<1, 1>>>(N, M, u, ldu);
    update_east_west_boundary<<<1, 1>>>(M, N, u, ldu);
    update_advection_field_kernel<<<1, 1>>>(M, N, &u[ldu + 1], ldu, &v[ldv + 1],
                                            ldv, Ux, Uy);
    copy_field_kernel<<<1, 1>>>(M, N, &v[ldv + 1], ldv, &u[ldu + 1], ldu);
  } // for(r...)
  HANDLE_ERROR(cudaFree(v));
} // run_serial_advection_device()
