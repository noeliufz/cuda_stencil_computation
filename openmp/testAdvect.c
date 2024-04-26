// OpenMP 2D advection solver test program

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt()
#include <assert.h>
#include <sys/time.h> //gettimeofday()
#include <stdbool.h> //bool type
#include <omp.h>

#include "serAdvect.h"
#include "parAdvect.h"

#define USAGE "OMP_NUM_THREADS=p testAdvect [-P P] [-x] [-v v] M N [r]"
#define DEFAULTS "P=p r=1 v=0"
#define OPTCHARS "P:xv:"

int M, N;               // advection field size
int P, Q;               // PxQ decomposition, Q = nprocs / P
int n_timesteps = 1;              // number of timesteps for the simulation
bool opt_P = false;     // set if -P specified
bool opt_extra = false; // set if -x specified
int verbosity = false;  // v, above
int nprocs;             // p, above

// print a usage message for this program and exit with a status of 1
void show_usage_message(char *msg) {
  printf("testAdvect: %s\n", msg);
  printf("usage: %s\n\tdefault values: %s\n", USAGE, DEFAULTS);
  fflush(stdout);
  exit(1);
}

void parse_command_line_arguments(int argc, char *argv[]) {
  extern char *optarg; // points to option argument (for -p option)
  extern int optind;   // index of last option parsed by getopt()
  extern int opterr;
  char optchar;        // option character returned my getopt()
  opterr = 0;          // suppress getopt() error message for invalid option
  P = nprocs;
  while ((optchar = getopt(argc, argv, OPTCHARS)) != -1) {
    // extract next option from the command line     
    switch (optchar) {
    case 'P':
      if (sscanf(optarg, "%d", &P) != 1) // invalid integer 
	show_usage_message("bad value for P");
      opt_P = true;
      break;
    case 'v':
      if (sscanf(optarg, "%d", &verbosity) != 1) // invalid integer 
	show_usage_message("bad value for v");
      break;
    case 'x':
      opt_extra = true;
      break;
    default:
      show_usage_message("unknown option");
      break;
    } //switch 
   } //while

  if (P == 0 || nprocs % P != 0)
    show_usage_message("number of threads must be a multiple of P");
  Q = nprocs / P;
  assert (Q > 0);

  if (optind < argc) {
    if (sscanf(argv[optind], "%d", &M) != 1) 
      show_usage_message("bad value for M");
  } else
    show_usage_message("missing M");
  N = M;
  if (optind+1 < argc)
    if (sscanf(argv[optind+1], "%d", &N) != 1) 
      show_usage_message("bad value for N");
  if (optind+2 < argc)
    if (sscanf(argv[optind+2], "%d", &n_timesteps) != 1) 
      show_usage_message("bad value for r");
} //getArgs()


static void print_average(char *name, double total, int nVals) {
  printf("%s %.3e\n", name, total / nVals);
}

//return wall time in seconds
static double Wtime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return(1.0*tv.tv_sec + 1.0e-6*tv.tv_usec);
}

int main(int argc, char** argv) {
  double *u; int ldu; //advection field
  double t, gflops; //time

  nprocs = omp_get_max_threads();
  parse_command_line_arguments(argc, argv);

  printf("Advection of a %dx%d global field on %d threads" 
	 " for %d steps.\n", M, N, nprocs, n_timesteps);
  if (opt_extra)
    printf("\tusing extra optimization methods\n");
  if (opt_P)
    printf("\tusing a %dx%d decomposition\n", P, Q);
  
  init_advection_parameters(M, N);  
  init_parallel_parameter_values(M, N, P, Q, verbosity);

  ldu = N+2;
  u = calloc((M+2)*ldu, sizeof(double));
  init_advection_field(0, 0, M, N, &u[ldu+1], ldu);
  if (verbosity > 1)
    print_advection_field(0, "init u", M, N, &u[ldu+1], ldu);

  t = Wtime();
  if (opt_extra)
    run_parallel_omp_advection_with_extra_opts(n_timesteps, u, ldu);
  else if (opt_P)    
    run_parallel_omp_advection_2D_decomposition(n_timesteps, u, ldu); 
  else
    run_parallel_omp_advection_1D_decomposition(n_timesteps, u, ldu);
  t = Wtime() - t;

  gflops = 1.0e-09 * advection_flops_per_element * M * N * n_timesteps;
  printf("Advection time %.2es, GFLOPs rate=%.2e (per core %.2e)\n",
	 t, gflops / t,  gflops / t / (P*Q)); 

  if (verbosity > 1)
    print_advection_field(0, "final u", M+2, N+2, u, ldu);
  print_average("Avg error of final field: ", 
	    compute_error_advection_field(n_timesteps, 0, 0, M, N, &u[ldu+1], ldu), M*N);
  print_average("Max error of final field: ", 
	    compute_max_error_advection_field(n_timesteps, 0, 0, M, N, &u[ldu+1], ldu), 1);

  free(u);
  return 0;
} //main()

