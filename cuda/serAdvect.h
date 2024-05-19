// serial 2D advection solver module
#include <string> //std::string

#define HANDLE_ERROR(err) (cudaHandleError(err, __FILE__, __LINE__))
void cudaHandleError(cudaError_t err, const char *file, int line);

// number of FLOPs to update a single element in the advection function
extern const int advection_flops_per_element;

// parameters needed for advection solvers
extern const double Velx, Vely; // advection velocity
extern double dt;               // time for 1 step
extern double deltax, deltay;   // grid spacing

// initializes the advection parameters for a global M x N field
void init_advection_parameters(int M, int N);

// access element (i,j) of array u with leading dimension ldu
#define V(u, i, j) u[(i) * (ld##u) + (j)]
#define V_(u, ldu, i, j) u[(i) * (ldu) + (j)]

// initialize (non-halo elements) of an M x N advection field (u, ldu)
void init_advection_field(int M, int N, double *u, int ldu);

// sum errors in an M x N advection field (u, ldu) after r timesteps
double compute_error_advection_field(int r, int M, int N, double *u, int ldu);

// get abs max error in an M x N advection field (u,ldu) after r timesteps
double compute_max_error_advection_field(int r, int M, int N, double *u,
                                         int ldu);

// print out the m x n advection field (u, ldu)
void print_advection_field(std::string label, int m, int n, double *u, int ldu);

#if 0 // for some reason, nvcc would not export this function properly
// calculate 1D coefficients for the advection stencil
__host__ __device__
void calculate_and_update_coefficients(double v, double *cm1, double *c0, double *cp1);
#endif

// update 1 timestep for the local advection, without updating halos
//   the M x N row-major array (v,ldv) is updated from (u,ldu)
//   Assumes a halo of width 1 are around this array;
//   the corners of the halo are at u[-1,-1], u[-1,n], u[m,-1] and u[m,n]
__host__ __device__ void update_advection_field(int M, int N, double *u,
                                                int ldu, double *v, int ldv,
                                                double Ux, double Uy);

// copy M x N field (v, ldv) to (u, ldu)
__host__ __device__ void copy_field(int M, int N, double *v, int ldv, double *u,
                                    int ldu);

// evolve advection on host over r timesteps, with (u,ldu) storing the field
void run_serial_advection_host(int M, int N, int r, double *u, int ldu);

// evolve advection on GPU over r timesteps, with (u,ldu) storing the field
void run_serial_advection_device(int M, int N, int r, double *u, int ldu);

// kernels that it uses
__global__ void update_east_west_boundary(int M, int N, double *u, int ldu);
__global__ void update_north_south_boundary(int N, int M, double *u, int ldu);
__global__ void update_advection_field_kernel(int M, int N, double *u, int ldu,
                                              double *v, int ldv, double Ux,
                                              double Uy);
__global__ void copy_field_kernel(int M, int N, double *v, int ldv, double *u,
                                  int ldu);
