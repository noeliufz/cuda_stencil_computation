// OpenMP parallel 2D advection solver module

#include "serAdvect.h" // advection parameters
#include <assert.h>
#include <omp.h>
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
#pragma omp parallel
  {
#pragma omp for
    for (j = 1; j < N + 1; j++) { // top and bottom halo
      u[j] = u[M * ldu + j];
      u[(M + 1) * ldu + j] = u[ldu + j];
    }
#pragma omp for
    for (i = 0; i < M + 2; i++) { // left and right sides of halo
      u[i * ldu] = u[i * ldu + N];
      u[i * ldu + N + 1] = u[i * ldu + 1];
    }
  }
}

void omp_update_advection_field_1D_decomposition(double *u, int ldu, double *v,
                                                 int ldv) {

  int i, j;
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);

  // if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
  //     fprintf(stderr, "PAPI library initialization error!\n");
  //     exit(1);
  // }

  // int event_set = PAPI_NULL;
  // if (PAPI_create_eventset(&event_set) != PAPI_OK) {
  //     fprintf(stderr, "PAPI create event set error!\n");
  //     exit(1);
  // }

  // if (PAPI_add_event(event_set, PAPI_L1_DCM) != PAPI_OK) {
  //     fprintf(stderr, "PAPI add event error!\n");
  //     exit(1);
  // }

  // if (PAPI_start(event_set) != PAPI_OK) {
  //     fprintf(stderr, "PAPI start error!\n");
  //     exit(1);
  // }
  // case 3
  // #pragma omp parallel for schedule(static, 1)
  //   for (j = 0; j < N; j++) {
  //     for (i = 0; i < M; i++) {
  //       // printf("N: %d\n", N);
  //       // printf("thread id %d visiting %d,%d\n", omp_get_thread_num(), i,
  //       j); v[i * ldv + j] =
  //           cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu +
  //           j] +
  //                   cjp1 * u[(i - 1) * ldu + j + 1]) +
  //           ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
  //                  cjp1 * u[i * ldu + j + 1]) +
  //           cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu +
  //           j] +
  //                   cjp1 * u[(i + 1) * ldu + j + 1]);
  //     }
  //   }

  // if (PAPI_stop(event_set, NULL) != PAPI_OK) {
  //     fprintf(stderr, "PAPI stop error!\n");
  //     exit(1);
  // }

  // long long values[2];
  // if (PAPI_read(event_set, values) != PAPI_OK) {
  //     fprintf(stderr, "PAPI read error!\n");
  //     exit(1);
  // }

  // printf("L1 Data Cache Misses: %lld\n", values[0]);
  // printf("L1 Data Cache Write Misses: %lld\n", values[1]);

  // if (PAPI_cleanup_eventset(event_set) != PAPI_OK) {
  //     fprintf(stderr, "PAPI cleanup event set error!\n");
  //     exit(1);
  // }
  // if (PAPI_destroy_eventset(&event_set) != PAPI_OK) {
  //     fprintf(stderr, "PAPI destroy event set error!\n");
  //     exit(1);
  // }
  // PAPI_shutdown();

// case 4
// #pragma omp parallel for private(i, j) schedule(static, 1)

// case 1
#pragma omp parallel for private(i, j) schedule(static)
  for (i = 0; i < M; i++) {
    // case 2
    // #pragma omp parallel for private(j)
    for (j = 0; j < N; j++) {
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
#pragma omp parallel
  {
    for (i = 0; i < M; i++)
#pragma omp parallel for
      for (j = 0; j < N; j++)
        out[i * ldout + j] = in[i * ldin + j];
  }
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
// using 2D parallelization
void run_parallel_omp_advection_2D_decomposition(int reps, double *u, int ldu) {
  int ldv = N + 2;

  double *v = calloc(ldv * (M + 2), sizeof(double));
  assert(v != NULL);
// Single parallel region
#pragma omp parallel default(shared)
  {
    int i, j;
    int r, P0, Q0;
    int M_loc, N_loc;
    int M0, N0;
    int id;

    id = omp_get_thread_num();

    P0 = id / Q;
    M0 = (M / P) * P0;
    M_loc = (P0 < P - 1) ? (M / P) : (M - M0);

    Q0 = id % Q;
    N0 = (N / Q) * Q0;
    N_loc = (Q0 < Q - 1) ? (N / Q) : (N - N0);

    // Fix rows to update
    int rows_to_update = P0 < P - 1 ? M_loc : M_loc + 2;

    for (r = 0; r < reps; r++) {
      // Update top and bottom
      for (j = N0 + 1; j < N0 + N_loc + 1; j++) {
        u[j] = u[ldu * M + j];
        u[ldu * (M + 1) + j] = u[ldu + j];
      }

      // printf("id: %d in rep: %d\n", id, r);
#pragma omp barrier

      // Update left and right
      for (i = M0; i < M0 + rows_to_update; i++) { // left and right  halo
        u[ldu * i] = u[ldu * i + N];
        u[ldu * i + N + 1] = u[ldu * i + 1];
      }

#pragma omp barrier

      // Update advection
      update_advection_field(M_loc, N_loc, &u[ldu * (M0 + 1) + N0 + 1], ldu,
                             &v[ldv * (M0 + 1) + N0 + 1], ldv);

#pragma omp barrier

      // copy back
      copy_field(M_loc, N_loc, &v[ldv * (M0 + 1) + N0 + 1], ldv,
                 &u[ldu * (M0 + 1) + N0 + 1], ldu);

    } // for (r...)
  }
  free(v);
} // run_parallel_omp_advection_2D_decomposition()

// extra optimization variant
void run_parallel_omp_advection_with_extra_opts(int reps, double *u, int ldu) {
} // run_parallel_omp_advection_with_extra_opts()
