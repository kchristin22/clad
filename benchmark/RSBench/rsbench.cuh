#define _CubLog
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<math.h>
#include<string.h>
#include<stdint.h>
#include<float.h>
#include<assert.h>
#include<cuda.h>
#include <thrust/reduce.h>
#include <chrono> 

#define PI 3.14159265359

// typedefs
typedef enum __hm{SMALL, LARGE, XL, XXL} HM_size;

#define HISTORY_BASED 1
#define EVENT_BASED 2

#define STARTING_SEED 1070
#define INITIALIZATION_SEED 42

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define non_differentiable __attribute__((annotate("non_differentiable")))

typedef struct{
	double r;
	double i;
} RSComplex;

namespace clad{
	namespace custom_derivatives{
		namespace class_functions{
            __host__ __device__ inline void constructor_pullback(RSComplex &other,
                                                         RSComplex *d_this,
                                                         RSComplex *d_other)
            {
                assert(d_this != nullptr && "d_this is NULL!");
                assert(d_other != nullptr && "d_other is NULL!");
                d_other->r += d_this->r;
                d_other->i += d_this->i;
            };
        }
	}
}

typedef struct{
	int nthreads;
	int n_nuclides;
	int lookups;
	HM_size HM;
	int avg_n_poles;
	int avg_n_windows;
	int numL;
	int doppler;
	int particles;
	int simulation_method;
	int kernel_id;
} non_differentiable Input;

typedef struct{
	RSComplex MP_EA;
	RSComplex MP_RT;
	RSComplex MP_RA;
	RSComplex MP_RF;
	short int l_value;
} Pole;

namespace clad
{
    namespace custom_derivatives
    {
        namespace class_functions
        {
            __device__ void constructor_pullback(Pole &other, Pole *d_this,
                                                 Pole *d_other)
            {
				constructor_pullback(other.MP_EA, &d_this->MP_EA, &d_other->MP_EA);
				constructor_pullback(other.MP_RT, &d_this->MP_RT, &d_other->MP_RT);
				constructor_pullback(other.MP_RA, &d_this->MP_RA, &d_other->MP_RA);
				d_other->l_value += d_this->l_value;
            };
        }
    }
}

typedef struct{
	double T;
	double A;
	double F;
	int start;
	int end;
} non_differentiable Window;

typedef struct{
	int * n_poles;
	unsigned long length_n_poles;
	int * n_windows;
	unsigned long length_n_windows;
	Pole * poles;
	Pole * d_poles;
	unsigned long length_poles;
	Window * windows;
	unsigned long length_windows;
	double * pseudo_K0RS;
	unsigned long length_pseudo_K0RS;
	int * num_nucs;
	unsigned long length_num_nucs;
	int * mats;
	unsigned long length_mats;
	double * concs;
	unsigned long length_concs;
	int max_num_nucs;
	int max_num_poles;
	int max_num_windows;
	double * p_energy_samples;
	unsigned long length_p_energy_samples;
	int * mat_samples;
	unsigned long length_mat_samples;
	unsigned long  * verification;
	unsigned long length_verification;
	double * dout;
} SimulationData;

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int( int a );
Input read_CLI( int argc, char * argv[] );
void print_CLI_error(void);
void print_input_summary(Input input);
int validate_and_print_results(Input input, double runtime, unsigned long vhash);

// init.c
SimulationData initialize_simulation( Input input );
int * generate_n_poles( Input input,  uint64_t * seed );
int * generate_n_windows( Input input ,  uint64_t * seed);
Pole * generate_poles( Input input, int * n_poles, uint64_t * seed, int * max_num_poles );
Window * generate_window_params( Input input, int * n_windows, int * n_poles, uint64_t * seed, int * max_num_windows );
double * generate_pseudo_K0RS( Input input, uint64_t * seed );
SimulationData move_simulation_data_to_device( Input in, SimulationData SD );

// material.c
int * load_num_nucs(Input input);
int * load_mats( Input input, int * num_nucs, int * max_num_nucs, unsigned long * length_mats );
double * load_concs( int * num_nucs, uint64_t * seed, int max_num_nucs );
SimulationData get_materials(Input input, uint64_t * seed);

// utils.c
size_t get_mem_estimate( Input input );
double get_time(void);

// simulation.c
void run_event_based_simulation(Input input, SimulationData data, SimulationData SD, unsigned long * vhash_result );
void run_event_based_simulation_optimization_1(Input in, SimulationData GSD, unsigned long * vhash_result);
__global__ void xs_lookup_kernel_baseline(Input in, SimulationData GSD );
__device__ void calculate_macro_xs( double * macro_xs, int mat, double E, Input input, const int * num_nucs, const int * mats, int max_num_nucs, const double * concs, const int * n_windows, const double * pseudo_K0Rs, const Window * windows, Pole * poles, int max_num_windows, int max_num_poles );
__device__ void calculate_micro_xs( double * micro_xs, int nuc, double E, Input input, const int * n_windows, const double * pseudo_K0RS, const Window * windows, Pole * poles, int max_num_windows, int max_num_poles);
__device__ void calculate_micro_xs_doppler( double * micro_xs, int nuc, double E, Input input, const int * n_windows, const double * pseudo_K0RS, const Window * windows, Pole * poles, int max_num_windows, int max_num_poles );
__device__ int pick_mat( uint64_t * seed );
__device__ void calculate_sig_T( int nuc, double E, Input input, const double * pseudo_K0RS, RSComplex * sigTfactors );
__device__ RSComplex fast_nuclear_W( RSComplex Z );
__host__ __device__ double LCG_random_double(uint64_t * seed);
__host__ __device__ uint64_t LCG_random_int(uint64_t * seed);
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
__device__ RSComplex c_add( RSComplex A, RSComplex B);
__device__ RSComplex c_sub( RSComplex A, RSComplex B);
__host__ __device__ RSComplex c_mul( RSComplex A, RSComplex B);
__device__ RSComplex c_div( RSComplex A, RSComplex B);
__device__ double c_abs( RSComplex A);
__device__ double fast_exp(double x);
__device__ RSComplex fast_cexp( RSComplex z );
