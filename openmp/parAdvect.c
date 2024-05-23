// OpenMP parallel 2D advection solver module

#include "serAdvect.h" // advection parameters
#include <assert.h>
#include <omp.h>
#include <papi.h>
#include <stdio.h>
#include <stdlib.h>

static int M, N, P, Q;
static int verbosity;

// sets up parameters above
void init_parallel_parameter_values(int M_, int N_, int P_, int Q_, int verb) {
  M = M_, N = N_;
  P = P_, Q = Q_;
  verbosity = verb;
} // init_parallel_parameter_values()

void omp_update_boundary_1D_decomposition(double *u, int ldu) {
  int i, j;
  for (j = 1; j < N + 1; j++) { // top and bottom halo
    u[j] = u[M * ldu + j];
    u[(M + 1) * ldu + j] = u[ldu + j];
  }
  for (i = 0; i < M + 2; i++) { // left and right sides of halo
    u[i * ldu] = u[i * ldu + N];
    u[i * ldu + N + 1] = u[i * ldu + 1];
  }
}

void omp_update_advection_field_1D_decomposition(double *u, int ldu, double *v,
                                                 int ldv) {

  int i, j;
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);

#pragma omp parallel for private(i, j)
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
} // omp_update_advection_field_1D_decomposition()

void omp_copy_field_1D_decomposition(double *in, int ldin, double *out,
                                     int ldout) {
  int i, j;
#pragma omp parallel for private(i, j)
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      out[i * ldout + j] = in[i * ldin + j];
} // omp_copy_field_1D_decomposition()

// evolve advection over reps timesteps, with (u,ldu) containing the field
// using 1D parallelization
void run_parallel_omp_advection_1D_decomposition(int reps, double *u, int ldu) {
  int r, ldv = N + 2;
  double *v = calloc(ldv * (M + 2), sizeof(double));
  assert(v != NULL);
  for (r = 0; r < reps; r++) {
    omp_update_boundary_1D_decomposition(u, ldu);
    omp_update_advection_field_1D_decomposition(&u[ldu + 1], ldu, &v[ldv + 1],
                                                ldv);
    omp_copy_field_1D_decomposition(&v[ldv + 1], ldv, &u[ldu + 1], ldu);
  } // for (r...)
  free(v);
} // run_parallel_omp_advection_1D_decomposition()

// ... using 2D parallelization
void run_parallel_omp_advection_2D_decomposition(int reps, double *u, int ldu) {
  int i, j;
  int r, ldv = N + 2;
  double *v = calloc(ldv * (M + 2), sizeof(double));
  assert(v != NULL);

  for (r = 0; r < reps; r++) {
    for (j = 1; j < N + 1; j++) { // top and bottom halo
      u[j] = u[M * ldu + j];
      u[(M + 1) * ldu + j] = u[ldu + j];
    }
    for (i = 0; i < M + 2; i++) { // left and right sides of halo
      u[i * ldu] = u[i * ldu + N];
      u[i * ldu + N + 1] = u[i * ldu + 1];
    }

    update_advection_field(M, N, &u[ldu + 1], ldu, &v[ldv + 1], ldv);

    copy_field(M, N, &v[ldv + 1], ldv, &u[ldu + 1], ldu);
  } // for (r...)
  free(v);
} // run_parallel_omp_advection_2D_decomposition()

// ... extra optimization variant
void run_parallel_omp_advection_with_extra_opts(int reps, double *u, int ldu) {

} // run_parallel_omp_advection_with_extra_opts()
