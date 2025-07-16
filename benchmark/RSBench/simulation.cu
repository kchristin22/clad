#include "rsbench.cuh"
#include "clad/Differentiator/Differentiator.h"

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// implementation of the algorithm, with only minor GPU optimizations in place.
// Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, RSBench will only run the baseline implementation. Optimized variants
// must be specifically selected using the "-k <optimized variant ID>" command
// line argument.
////////////////////////////////////////////////////////////////////////////////////

__attribute__((device)) void calculate_macro_xs_grad_0_11(
    double *__restrict macro_xs, int mat, double E, Input input,
    const int *__restrict num_nucs, const int *__restrict mats,
    int max_num_nucs, const double *__restrict concs,
    const int *__restrict n_windows, const double *__restrict pseudo_K0Rs,
    const Window *__restrict windows, Pole *__restrict poles,
    int max_num_windows, int max_num_poles, double *_d_macro_xs,
    Pole *_d_poles);

void run_event_based_simulation(Input input, SimulationData GSD, SimulationData SD, unsigned long * vhash_result )
{
	////////////////////////////////////////////////////////////////////////////////
	// Configure & Launch Simulation Kernel
	////////////////////////////////////////////////////////////////////////////////
	printf("Running baseline event-based simulation on device...\n");

	int nthreads = 32;
	int nblocks = ceil( (double) input.lookups / 32.0);

	gpuErrchk( cudaDeviceSynchronize() );

	xs_lookup_kernel_baseline<<<nblocks, nthreads>>>( input, GSD );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	
	////////////////////////////////////////////////////////////////////////////////
	// Reduce Verification Results
	////////////////////////////////////////////////////////////////////////////////
	printf("Reducing verification results...\n");

	unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + input.lookups, 0);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	*vhash_result = verification_scalar;

	#ifdef VERIFY
	#if FORWARD

	double *dout = (double*)malloc(sizeof(double));
	gpuErrchk( cudaMemcpy(dout, GSD.dout, sizeof(double), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	printf("dout=%f\n", *dout);

	#else

	size_t sz = 10;//GSD.length_poles;
    Pole here[sz];// = 1.23456;
    gpuErrchk( cudaMemcpy(&here, GSD.d_poles, sz * sizeof(Pole), cudaMemcpyDeviceToHost) );

	for (int i=0; i < sz; ++i)
 	   printf("here[%d]=%f %f %f %f %f %f %f %f\n", i, here[i].MP_EA.r, here[i].MP_EA.i, here[i].MP_RT.r, here[i].MP_RT.i, here[i].MP_RA.r, here[i].MP_RA.i, here[i].MP_RF.r, here[i].MP_RF.i);

	#endif
	#endif
}

// In this kernel, we perform a single lookup with each thread. Threads within a warp
// do not really have any relation to each other, and divergence due to high nuclide count fuel
// material lookups are costly. This kernel constitutes baseline performance.
__global__ void xs_lookup_kernel_baseline(Input in, SimulationData GSD )
{
	// The lookup ID. Used to set the seed, and to store the verification value
	const int i = blockIdx.x *blockDim.x + threadIdx.x;

	if( i >= in.lookups )
		return;

	// Set the initial seed value
	uint64_t seed = STARTING_SEED;	

	// Forward seed to lookup index (we need 2 samples per lookup)
	seed = fast_forward_LCG(seed, 2*i);

	// Randomly pick an energy and material for the particle
	double E = LCG_random_double(&seed);
	int mat  = pick_mat(&seed);

	double macro_xs[4] = {0};

	#ifdef FORWARD
	calculate_macro_xs( macro_xs, mat, E, in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs, GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows,   GSD.poles, GSD.max_num_windows, GSD.max_num_poles );

	#ifdef VERIFY
	double macro_xs2[4] = {0};
	calculate_macro_xs( macro_xs2, mat, E, in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs, GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows, GSD.d_poles, GSD.max_num_windows, GSD.max_num_poles );
	printf("zz=%f %f %f %f\n", macro_xs2[0], macro_xs[0], GSD.poles[0].MP_EA.r , GSD.d_poles[0].MP_EA.r  );
	atomicAdd(GSD.dout, (macro_xs2[0] - macro_xs[0]) / 1e-3 );
	#endif

    #else

	double d_macro_xs[4] = {0};
	d_macro_xs[0] = 1.0;

    // auto grad = clad::gradient(calculate_macro_xs, "macro_xs, poles");
    // grad.execute(macro_xs, mat, E, in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs,
    //              GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows,
    //              GSD.poles, GSD.max_num_windows, GSD.max_num_poles,
    //              d_macro_xs, GSD.d_poles);
    calculate_macro_xs_grad_0_11(
        macro_xs, mat, E, in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs,
        GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows, GSD.poles,
        GSD.max_num_windows, GSD.max_num_poles, d_macro_xs, GSD.d_poles);

#endif

	// For verification, and to prevent the compiler from optimizing
	// all work out, we interrogate the returned macro_xs_vector array
	// to find its maximum value index, then increment the verification
	// value by that index. In this implementation, we write to a global
	// verification array that will get reduced after this kernel comples.
	double max = -DBL_MAX;
	int max_idx = 0;
	for(int x = 0; x < 4; x++ )
	{
		if( macro_xs[x] > max )
		{
			max = macro_xs[x];
			max_idx = x;
		}
	}
	GSD.verification[i] = max_idx+1;
}

__attribute__((noinline))
__device__ void body( int i, double * __restrict__ macro_xs, int mat, double E, const Input& __restrict__ input, int * __restrict__ num_nucs, int * __restrict__ mats, int max_num_nucs, double * __restrict__ concs, int * __restrict__ n_windows, double * __restrict__ pseudo_K0Rs, Window * __restrict__ windows, Pole * __restrict__ poles, int max_num_windows, int max_num_poles ) {
	double micro_xs[4];
	int nuc = mats[mat * max_num_nucs + i];

	if( input.doppler == 1 )
		calculate_micro_xs_doppler( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);
	else
		calculate_micro_xs( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);

	for( int j = 0; j < 4; j++ )
	{
		macro_xs[j] += micro_xs[j] * concs[mat * max_num_nucs + i];
	}	
}

__device__ void calculate_macro_xs( double * __restrict__ macro_xs, int mat, double E, Input input, const int * __restrict__ num_nucs, const int * __restrict__ mats, int max_num_nucs, const double * __restrict__ concs, const int * __restrict__ n_windows, const double * __restrict__ pseudo_K0Rs, const Window * __restrict__ windows, Pole * __restrict__ poles, int max_num_windows, int max_num_poles ) 
{
	// zero out macro vector
	// for( int i = 0; i < 4; i++ )
	// 	macro_xs[i] = 0;

	// for nuclide in mat
	int sz = num_nucs[mat];
	for( int i = 0; i < sz; i++ )
	// for( int i = 0; i < num_nucs[mat]; i++ )
	{
		//body(i, macro_xs, mat, E, input, num_nucs, mats, max_num_nucs, concs, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);
		
		///*
		double micro_xs[4];
		int nuc = mats[mat * max_num_nucs + i];

		if( input.doppler == 1 )
			calculate_micro_xs_doppler( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);
		else
			calculate_micro_xs( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);

		for( int j = 0; j < 4; j++ )
		{
			macro_xs[j] += micro_xs[j] * concs[mat * max_num_nucs + i];
		}
		//*/
		
		// Debug
		/*
		printf("E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\n",
		E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );
		*/
	}

	// Debug
	/*
	printf("E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\n",
	E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );
	*/
}

// No Temperature dependence (i.e., 0K evaluation)
#ifdef ALWAYS_INLINE
__attribute__((always_inline))
#endif
__device__ void calculate_micro_xs( double * micro_xs, int nuc, double E, Input input, const int * n_windows, const double * pseudo_K0RS, const Window * windows, Pole * poles, int max_num_windows, int max_num_poles)
{
	// MicroScopic XS's to Calculate
	double sigT;
	double sigA;
	double sigF;
	double sigE;

	// Calculate Window Index
	double spacing = 1.0 / n_windows[nuc];
	int window = (int) ( E / spacing );
	if( window == n_windows[nuc] )
		window--;

	// Calculate sigTfactors
	RSComplex sigTfactors[4]; // Of length input.numL, which is always 4
	calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors );

	// Calculate contributions from window "background" (i.e., poles outside window (pre-calculated)
	Window w = windows[nuc * max_num_windows + window];
	sigT = E * w.T;
	sigA = E * w.A;
	sigF = E * w.F;

	// Loop over Poles within window, add contributions
	for( int i = w.start; i < w.end; i++ )
	{
		RSComplex PSIIKI = {0., 0.};
        RSComplex CDUM = {0., 0.};
        Pole pole = poles[nuc * max_num_poles + i];
		RSComplex t1 = {0, 1};
		RSComplex t2 = {sqrt(E), 0 };
		PSIIKI = c_div( t1 , c_sub(pole.MP_EA,t2) );
		RSComplex E_c = {E, 0};
		CDUM = c_div(PSIIKI, E_c);
        RSComplex _t1 =
            c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value]));
        sigT += _t1.r;
        RSComplex _t2 = c_mul(pole.MP_RA, CDUM);
        sigA += _t2.r;
        RSComplex _t3 = c_mul(pole.MP_RF, CDUM);
        sigF += _t3.r;
    }

	sigE = sigT - sigA;

	micro_xs[0] = sigT;
	micro_xs[1] = sigA;
	micro_xs[2] = sigF;
	micro_xs[3] = sigE;
	// printf("reg %f %f %f %f\n", sigT, sigA, sigF, sigE);
}

// Temperature Dependent Variation of Kernel
// (This involves using the Complex Faddeeva function to
// Doppler broaden the poles within the window)
#ifdef ALWAYS_INLINE
__attribute__((always_inline))
#endif
__device__ void calculate_micro_xs_doppler( double * micro_xs, int nuc, double E, Input input, const int * n_windows, const double * pseudo_K0RS, const Window * windows, Pole * poles, int max_num_windows, int max_num_poles )
{
	// MicroScopic XS's to Calculate
	double sigT;
	double sigA;
	double sigF;
	double sigE;

	// Calculate Window Index
	double spacing = 1.0 / n_windows[nuc];
	int window = (int) ( E / spacing );
	if( window == n_windows[nuc] )
		window--;

	// Calculate sigTfactors
	RSComplex sigTfactors[4]; // Of length input.numL, which is always 4
	calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors );

	// Calculate contributions from window "background" (i.e., poles outside window (pre-calculated)
	Window w = windows[nuc * max_num_windows + window];
	sigT = E * w.T;
	sigA = E * w.A;
	sigF = E * w.F;

	double dopp = 0.5;

	//if (w.start == 0)
	//	printf("start=%d\n", w.start);
	// Loop over Poles within window, add contributions
	for( int i = w.start; i < w.end; i++ )
	{
		Pole pole = poles[nuc * max_num_poles + i];

		// Prep Z
		RSComplex E_c = {E, 0};
		RSComplex dopp_c = {dopp, 0};
		RSComplex Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);

		// Evaluate Fadeeva Function
		RSComplex faddeeva = fast_nuclear_W( Z );

		// Update W
        RSComplex _t1 =
            c_mul(pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]));
        sigT += _t1.r;
        RSComplex _t2 = c_mul(pole.MP_RA, faddeeva);
        sigA += _t2.r;
        RSComplex _t3 = c_mul(pole.MP_RF, faddeeva);
        sigF += _t3.r;
    }

	sigE = sigT - sigA;

	micro_xs[0] = sigT;
	micro_xs[1] = sigA;
	micro_xs[2] = sigF;
	micro_xs[3] = sigE;
}

// picks a material based on a probabilistic distribution
__device__ int pick_mat( uint64_t * seed )
{
	// I have a nice spreadsheet supporting these numbers. They are
	// the fractions (by volume) of material in the core. Not a 
	// *perfect* approximation of where XS lookups are going to occur,
	// but this will do a good job of biasing the system nonetheless.

	double dist[12];
	dist[0]  = 0.140;	// fuel
	dist[1]  = 0.052;	// cladding
	dist[2]  = 0.275;	// cold, borated water
	dist[3]  = 0.134;	// hot, borated water
	dist[4]  = 0.154;	// RPV
	dist[5]  = 0.064;	// Lower, radial reflector
	dist[6]  = 0.066;	// Upper reflector / top plate
	dist[7]  = 0.055;	// bottom plate
	dist[8]  = 0.008;	// bottom nozzle
	dist[9]  = 0.015;	// top nozzle
	dist[10] = 0.025;	// top of fuel assemblies
	dist[11] = 0.013;	// bottom of fuel assemblies

	double roll = LCG_random_double(seed);

	// makes a pick based on the distro
	for( int i = 0; i < 12; i++ )
	{
		double running = 0;
		for( int j = i; j > 0; j-- )
			running += dist[j];
		if( roll < running )
			return i;
	}

	return 0;
}

__device__ void calculate_sig_T( int nuc, double E, Input input, const double * pseudo_K0RS, RSComplex * sigTfactors )
{
	double phi;

	// #pragma unroll
	for( int i = 0; i < 4; i++ )
	{
		phi = pseudo_K0RS[nuc * input.numL + i] * sqrt(E);

		if( i == 1 )
			phi -= - atan( phi );
		else if( i == 2 )
			phi -= atan( 3.0 * phi / (3.0 - phi*phi));
		else if( i == 3 )
			phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));

		phi *= 2.0;

		sigTfactors[i].r = +cos(phi);
		sigTfactors[i].i = -sin(phi);
	}
}

// This function uses a combination of the Abrarov Approximation
// and the QUICK_W three term asymptotic expansion.
// Only expected to use Abrarov ~0.5% of the time.
__attribute__((always_inline))
__device__ RSComplex fast_nuclear_W( RSComplex Z )
{
	// Abrarov 
	if( c_abs(Z) < 6.0 )
	{
		// Precomputed parts for speeding things up
		// (N = 10, Tm = 12.0)
		RSComplex prefactor = {0, 8.124330e+01};
		double an[10] = {
			2.758402e-01,
			2.245740e-01,
			1.594149e-01,
			9.866577e-02,
			5.324414e-02,
			2.505215e-02,
			1.027747e-02,
			3.676164e-03,
			1.146494e-03,
			3.117570e-04
		};
		double neg_1n[10] = {
			-1.0,
			1.0,
			-1.0,
			1.0,
			-1.0,
			1.0,
			-1.0,
			1.0,
			-1.0,
			1.0
		};

		double denominator_left[10] = {
			9.869604e+00,
			3.947842e+01,
			8.882644e+01,
			1.579137e+02,
			2.467401e+02,
			3.553058e+02,
			4.836106e+02,
			6.316547e+02,
			7.994380e+02,
			9.869604e+02
		};

		RSComplex t1 = {0, 12};
		RSComplex t2 = {12, 0};
		RSComplex i = {0,1};
		RSComplex one = {1, 0};
		RSComplex W = c_div(c_mul(i, ( c_sub(one, fast_cexp(c_mul(t1, Z))) )) , c_mul(t2, Z));
		RSComplex sum = {0,0};
		// #pragma unroll
		for( int n = 0; n < 10; n++ )
		{
			RSComplex t3 = {neg_1n[n], 0};
			RSComplex top = c_sub(c_mul(t3, fast_cexp(c_mul(t1, Z))), one);
			RSComplex t4 = {denominator_left[n], 0};
			RSComplex t5 = {144, 0};
			RSComplex bot = c_sub(t4, c_mul(t5,c_mul(Z,Z)));
			RSComplex t6 = {an[n], 0};
			sum = c_add(sum, c_mul(t6, c_div(top,bot)));
		}
		W = c_add(W, c_mul(prefactor, c_mul(Z, sum)));
		return W;
	}
	else
	{
		// QUICK_2 3 Term Asymptotic Expansion (Accurate to O(1e-6)).
		// Pre-computed parameters
		RSComplex a = {0.512424224754768462984202823134979415014943561548661637413182,0};
		RSComplex b = {0.275255128608410950901357962647054304017026259671664935783653, 0};
		RSComplex c = {0.051765358792987823963876628425793170829107067780337219430904, 0};
		RSComplex d = {2.724744871391589049098642037352945695982973740328335064216346, 0};

		RSComplex i = {0,1};
		RSComplex Z2 = c_mul(Z, Z);
		// Three Term Asymptotic Expansion
		RSComplex W = c_mul(c_mul(Z,i), (c_add(c_div(a,(c_sub(Z2, b))) , c_div(c,(c_sub(Z2, d))))));

		return W;
	}
}

__host__ __device__ double LCG_random_double(uint64_t * seed)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}	

__host__ __device__ uint64_t LCG_random_int(uint64_t * seed)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return *seed;
}	

__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;

	n = n % m;

	uint64_t a_new = 1;
	uint64_t c_new = 0;

	while(n > 0) 
	{
		if(n & 1)
		{
			a_new *= a;
			c_new = c_new * a + c;
		}
		c *= (a + 1);
		a *= a;

		n >>= 1;
	}

	return (a_new * seed + c_new) % m;
}

// Complex arithmetic functions

__device__ RSComplex c_add( RSComplex A, RSComplex B)
{
	RSComplex C;
	C.r = A.r + B.r;
	C.i = A.i + B.i;
	return C;
}

__device__ RSComplex c_sub( RSComplex A, RSComplex B)
{
	RSComplex C;
	C.r = A.r - B.r;
	C.i = A.i - B.i;
	return C;
}

__host__ __device__ RSComplex c_mul( RSComplex A, RSComplex B)
{
	double a = A.r;
	double b = A.i;
	double c = B.r;
	double d = B.i;
	RSComplex C;
	C.r = (a*c) - (b*d);
	C.i = (a*d) + (b*c);
	return C;
}

__device__ RSComplex c_div( RSComplex A, RSComplex B)
{
	double a = A.r;
	double b = A.i;
	double c = B.r;
	double d = B.i;
	RSComplex C;
	double denom = c*c + d*d;
	C.r = ( (a*c) + (b*d) ) / denom;
	C.i = ( (b*c) - (a*d) ) / denom;
	return C;
}

__device__ double c_abs( RSComplex A)
{
	return sqrt(A.r*A.r + A.i * A.i);
}


// Fast (but inaccurate) exponential function
// Written By "ACMer":
// https://codingforspeed.com/using-faster-exponential-approximation/
// We use our own to avoid small differences in compiler specific
// exp() intrinsic implementations that make it difficult to verify
// if the code is working correctly or not.
__device__ double fast_exp(double x)
{
  x = 1.0 + x * 0.000244140625;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  return x;
}

// Implementation based on:
// z = x + iy
// cexp(z) = e^x * (cos(y) + i * sin(y))
__device__ RSComplex fast_cexp( RSComplex z )
{
	double x = z.r;
	double y = z.i;

	// For consistency across architectures, we
	// will use our own exponetial implementation
	//double t1 = exp(x);
	double t1 = fast_exp(x);
	double t2 = cos(y);
	double t3 = sin(y);
	RSComplex t4 = {t2, t3};
	RSComplex t5 = {t1, 0};
	RSComplex result = c_mul(t5, (t4));
	return result;
}	

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
// OPTIMIZED VARIANT FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// This section contains a number of optimized variants of some of the above
// functions, which each deploy a different combination of optimizations strategies
// specific to GPU. By default, RSBench will not run any of these variants. They
// must be specifically selected using the "-k <optimized variant ID>" command
// line argument.
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////
// Optimization 6 -- Kernel Splitting + All Material Lookups + Full Sort
//                   + Energy Sort
////////////////////////////////////////////////////////////////////////////////////
// This optimization builds on optimization 4, adding in a second sort by energy.
// It is extremely fast, as now most of the threads within a warp will be hitting
// the same indices in the lookup grids. This greatly reduces thread divergence and
// greatly improves cache efficiency and re-use.
//
// However, it is unlikely that this exact optimization would be possible in a real
// application like OpenMC. One major difference is that particle objects are quite
// large, often having 50+ variable fields, such that sorting them in memory becomes
// rather expensive. Instead, the best possible option would probably be to create
// intermediate indexing (per Hamilton et. al 2019), and run the kernels indirectly.
////////////////////////////////////////////////////////////////////////////////////

__global__ void sampling_kernel(Input in, SimulationData GSD )
{
	// The lookup ID.
	const int i = blockIdx.x *blockDim.x + threadIdx.x;

	if( i >= in.lookups )
		return;

	// Set the initial seed value
	uint64_t seed = STARTING_SEED;	

	// Forward seed to lookup index (we need 2 samples per lookup)
	seed = fast_forward_LCG(seed, 2*i);

	// Randomly pick an energy and material for the particle
	double p_energy = LCG_random_double(&seed);
	int mat         = pick_mat(&seed); 

	// Store sample data in state array
	GSD.p_energy_samples[i] = p_energy;
	GSD.mat_samples[i] = mat;
}

__global__ void xs_lookup_kernel_optimization_1(Input in, SimulationData GSD, int m, int n_lookups, int offset )
{
	// The lookup ID. Used to set the seed, and to store the verification value
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if( i >= n_lookups )
		return;

	i += offset;

	// Check that our material type matches the kernel material
	int mat = GSD.mat_samples[i];
	if( mat != m )
		return;
	
	double macro_xs[4] = {0};

	calculate_macro_xs( macro_xs, mat, GSD.p_energy_samples[i], in, GSD.num_nucs, GSD.mats, GSD.max_num_nucs, GSD.concs, GSD.n_windows, GSD.pseudo_K0RS, GSD.windows, GSD.poles, GSD.max_num_windows, GSD.max_num_poles );

	// For verification, and to prevent the compiler from optimizing
	// all work out, we interrogate the returned macro_xs_vector array
	// to find its maximum value index, then increment the verification
	// value by that index. In this implementation, we write to a global
	// verification array that will get reduced after this kernel comples.
	double max = -DBL_MAX;
	int max_idx = 0;
	for(int x = 0; x < 4; x++ )
	{
		if( macro_xs[x] > max )
		{
			max = macro_xs[x];
			max_idx = x;
		}
	}
	GSD.verification[i] = max_idx+1;
}

void run_event_based_simulation_optimization_1(Input in, SimulationData GSD, unsigned long * vhash_result)
{
	const char * optimization_name = "Optimization 1 - Material & Energy Sorts + Material-specific Kernels";
	
	printf("Simulation Kernel:\"%s\"\n", optimization_name);
	
	////////////////////////////////////////////////////////////////////////////////
	// Allocate Additional Data Structures Needed by Optimized Kernel
	////////////////////////////////////////////////////////////////////////////////
	printf("Allocating additional device data required by kernel...\n");
	size_t sz;
	size_t total_sz = 0;

	sz = in.lookups * sizeof(double);
	gpuErrchk( cudaMalloc((void **) &GSD.p_energy_samples, sz) );
	total_sz += sz;
	GSD.length_p_energy_samples = in.lookups;

	sz = in.lookups * sizeof(int);
	gpuErrchk( cudaMalloc((void **) &GSD.mat_samples, sz) );
	total_sz += sz;
	GSD.length_mat_samples = in.lookups;
	
	printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0);

	////////////////////////////////////////////////////////////////////////////////
	// Configure & Launch Simulation Kernel
	////////////////////////////////////////////////////////////////////////////////
	printf("Beginning optimized simulation...\n");

	int nthreads = 32;
	int nblocks = ceil( (double) in.lookups / 32.0);
	
	sampling_kernel<<<nblocks, nthreads>>>( in, GSD );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// Count the number of fuel material lookups that need to be performed (fuel id = 0)
	int n_lookups_per_material[12];
	for( int m = 0; m < 12; m++ )
		n_lookups_per_material[m] = thrust::count(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, m);

	// Sort by material first
	thrust::sort_by_key(thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, GSD.p_energy_samples);

	// Now, sort each material by energy
	int offset = 0;
	for( int m = 0; m < 12; m++ )
	{
		thrust::sort_by_key(thrust::device, GSD.p_energy_samples + offset, GSD.p_energy_samples + offset + n_lookups_per_material[m], GSD.mat_samples + offset);
		offset += n_lookups_per_material[m];
	}
	
	// Launch all material kernels individually
	offset = 0;
	for( int m = 0; m < 12; m++ )
	{
		nthreads = 32;
		nblocks = ceil((double) n_lookups_per_material[m] / (double) nthreads);
		xs_lookup_kernel_optimization_1<<<nblocks, nthreads>>>( in, GSD, m, n_lookups_per_material[m], offset );
		offset += n_lookups_per_material[m];
	}
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	
	////////////////////////////////////////////////////////////////////////////////
	// Reduce Verification Results
	////////////////////////////////////////////////////////////////////////////////
	printf("Reducing verification results...\n");

	unsigned long verification_scalar = thrust::reduce(thrust::device, GSD.verification, GSD.verification + in.lookups, 0);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	*vhash_result = verification_scalar;
}

static inline constexpr void constructor_pullback(const Window &arg, Window *_d_this, Window *_d_arg) noexcept {
    {
        (*_d_arg).end += _d_this->end;
        _d_this->end = 0;
    }
    {
        (*_d_arg).start += _d_this->start;
        _d_this->start = 0;
    }
    {
        (*_d_arg).F += _d_this->F;
        _d_this->F = 0.;
    }
    {
        (*_d_arg).A += _d_this->A;
        _d_this->A = 0.;
    }
    {
        (*_d_arg).T += _d_this->T;
        _d_this->T = 0.;
    }
}

__attribute__((device)) void c_mul_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A, RSComplex *_d_B) {
    double _d_a = 0.;
    double a = A.r;
    double _d_b = 0.;
    double b = A.i;
    double _d_c = 0.;
    double c = B.r;
    double _d_d = 0.;
    double d = B.i;
    RSComplex _d_C = {0., 0.};
    RSComplex C;
    C.r = (a * c) - (b * d);
    C.i = (a * d) + (b * c);
    clad::custom_derivatives::class_functions::constructor_pullback(std::move(C), &_d_y, &_d_C);
    {
        double _r_d1 = _d_C.i;
        _d_C.i = 0.;
        _d_a += _r_d1 * d;
        _d_d += a * _r_d1;
        _d_b += _r_d1 * c;
        _d_c += b * _r_d1;
    }
    {
        double _r_d0 = _d_C.r;
        _d_C.r = 0.;
        _d_a += _r_d0 * c;
        _d_c += a * _r_d0;
        _d_b += -_r_d0 * d;
        _d_d += b * -_r_d0;
    }
    (*_d_B).i += _d_d;
    (*_d_B).r += _d_c;
    (*_d_A).i += _d_b;
    (*_d_A).r += _d_a;
}

__attribute__((device)) void c_sub_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A, RSComplex *_d_B) {
    RSComplex _d_C = {0., 0.};
    RSComplex C;
    C.r = A.r - B.r;
    C.i = A.i - B.i;
    clad::custom_derivatives::class_functions::constructor_pullback(std::move(C), &_d_y, &_d_C);
    {
        double _r_d1 = _d_C.i;
        _d_C.i = 0.;
        (*_d_A).i += _r_d1;
        (*_d_B).i += -_r_d1;
    }
    {
        double _r_d0 = _d_C.r;
        _d_C.r = 0.;
        (*_d_A).r += _r_d0;
        (*_d_B).r += -_r_d0;
    }
}
__attribute__((device)) void c_abs_pullback(RSComplex A, double _d_y, RSComplex *_d_A) {
    {
        double _r0 = 0.;
        _r0 += _d_y * clad::custom_derivatives::std::sqrt_pushforward(A.r * A.r + A.i * A.i, 1.).pushforward;
        (*_d_A).r += _r0 * A.r;
        (*_d_A).r += A.r * _r0;
        (*_d_A).i += _r0 * A.i;
        (*_d_A).i += A.i * _r0;
    }
}
__attribute__((device)) void c_div_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A, RSComplex *_d_B) {
    double _d_a = 0.;
    double a = A.r;
    double _d_b = 0.;
    double b = A.i;
    double _d_c = 0.;
    double c = B.r;
    double _d_d = 0.;
    double d = B.i;
    RSComplex _d_C = {0., 0.};
    RSComplex C;
    double _d_denom = 0.;
    double denom = c * c + d * d;
    C.r = ((a * c) + (b * d)) / denom;
    C.i = ((b * c) - (a * d)) / denom;
    clad::custom_derivatives::class_functions::constructor_pullback(std::move(C), &_d_y, &_d_C);
    {
        double _r_d1 = _d_C.i;
        _d_C.i = 0.;
        _d_b += _r_d1 / denom * c;
        _d_c += b * _r_d1 / denom;
        _d_a += -_r_d1 / denom * d;
        _d_d += a * -_r_d1 / denom;
        double _r1 = _r_d1 * -(((b * c) - (a * d)) / (denom * denom));
        _d_denom += _r1;
    }
    {
        double _r_d0 = _d_C.r;
        _d_C.r = 0.;
        _d_a += _r_d0 / denom * c;
        _d_c += a * _r_d0 / denom;
        _d_b += _r_d0 / denom * d;
        _d_d += b * _r_d0 / denom;
        double _r0 = _r_d0 * -(((a * c) + (b * d)) / (denom * denom));
        _d_denom += _r0;
    }
    {
        _d_c += _d_denom * c;
        _d_c += c * _d_denom;
        _d_d += _d_denom * d;
        _d_d += d * _d_denom;
    }
    (*_d_B).i += _d_d;
    (*_d_B).r += _d_c;
    (*_d_A).i += _d_b;
    (*_d_A).r += _d_a;
}
__attribute__((device)) clad::ValueAndPushforward<double, double> fast_exp_pushforward(double x, double _d_x) {
    _d_x = 0. + _d_x * 2.44140625E-4 + x * 0.;
    x = 1. + x * 2.44140625E-4;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    _d_x = _d_x * x + x * _d_x;
    x *= x;
    return {x, _d_x};
}

__attribute__((device)) void fast_cexp_pullback(RSComplex z, RSComplex _d_y, RSComplex *_d_z) {
    double _d_x = 0.;
    double x = z.r;
    double _d_y0 = 0.;
    double y = z.i;
    double _d_t1 = 0.;
    double t1 = fast_exp(x);
    double _d_t2 = 0.;
    double t2 = cos(y);
    double _d_t3 = 0.;
    double t3 = sin(y);
    RSComplex _d_t4 = {0., 0.};
    RSComplex t4 = {t2, t3};
    RSComplex _d_t5 = {0., 0.};
    RSComplex t5 = {t1, 0};
    RSComplex _d_result = {0., 0.};
    RSComplex result = c_mul(t5, t4);
    clad::custom_derivatives::class_functions::constructor_pullback(std::move(result), &_d_y, &_d_result);
    {
        RSComplex _r3 = {0., 0.};
        RSComplex _r4 = {0., 0.};
        c_mul_pullback(t5, t4, _d_result, &_r3, &_r4);
        clad::custom_derivatives::class_functions::constructor_pullback(t5, &_r3, &_d_t5);
        clad::custom_derivatives::class_functions::constructor_pullback(t4, &_r4, &_d_t4);
    }
    _d_t1 += _d_t5.r;
    {
        _d_t2 += _d_t4.r;
        _d_t3 += _d_t4.i;
    }
    {
        double _r2 = 0.;
        _r2 += _d_t3 * clad::custom_derivatives::std::sin_pushforward(y, 1.).pushforward;
        _d_y0 += _r2;
    }
    {
        double _r1 = 0.;
        _r1 += _d_t2 * clad::custom_derivatives::std::cos_pushforward(y, 1.).pushforward;
        _d_y0 += _r1;
    }
    {
        double _r0 = 0.;
        _r0 += _d_t1 * fast_exp_pushforward(x, 1.).pushforward;
        _d_x += _r0;
    }
    (*_d_z).i += _d_y0;
    (*_d_z).r += _d_x;
}
__device__ inline constexpr clad::ValueAndAdjoint<RSComplex &, RSComplex &>
operator_equal_reverse_forw(RSComplex &this_1, RSComplex &&arg,
                            RSComplex *_d_this, RSComplex &&_d_arg) noexcept
{
    this_1.r = static_cast<RSComplex &&>(arg).r;
    this_1.i = static_cast<RSComplex &&>(arg).i;
    return {this_1, *_d_this};
}
__device__ inline constexpr void
operator_equal_pullback(RSComplex &this_1, RSComplex &&arg, RSComplex _d_y,
                        RSComplex *_d_this, RSComplex *_d_arg) noexcept
{
    double _t0 = this_1.r;
    this_1.r = static_cast<RSComplex &&>(arg).r;
    double _t1 = this_1.i;
    this_1.i = static_cast<RSComplex &&>(arg).i;
    {
        this_1.i = _t1;
        double _r_d1 = _d_this->i;
        _d_this->i = 0.;
        (*_d_arg).i += _r_d1;
    }
    {
        this_1.r = _t0;
        double _r_d0 = _d_this->r;
        _d_this->r = 0.;
        (*_d_arg).r += _r_d0;
    }
}
__attribute__((device)) void c_add_pullback(RSComplex A, RSComplex B, RSComplex _d_y, RSComplex *_d_A, RSComplex *_d_B) {
    RSComplex _d_C = {0., 0.};
    RSComplex C;
    C.r = A.r + B.r;
    C.i = A.i + B.i;
    clad::custom_derivatives::class_functions::constructor_pullback(std::move(C), &_d_y, &_d_C);
    {
        double _r_d1 = _d_C.i;
        _d_C.i = 0.;
        (*_d_A).i += _r_d1;
        (*_d_B).i += _r_d1;
    }
    {
        double _r_d0 = _d_C.r;
        _d_C.r = 0.;
        (*_d_A).r += _r_d0;
        (*_d_B).r += _r_d0;
    }
}
__attribute__((always_inline)) __attribute__((device)) void fast_nuclear_W_pullback(RSComplex Z, RSComplex _d_y, RSComplex *_d_Z) {
    bool _cond0;
    RSComplex _d_prefactor = {0., 0.};
    RSComplex prefactor = {0., 0.};
    double _d_an[10] = {0};
    clad::array<double> an(10UL);
    double _d_neg_1n[10] = {0};
    clad::array<double> neg_1n(10UL);
    double _d_denominator_left[10] = {0};
    clad::array<double> denominator_left(10UL);
    RSComplex _d_t1 = {0., 0.};
    RSComplex t1 = {0., 0.};
    RSComplex _d_t2 = {0., 0.};
    RSComplex t2 = {0., 0.};
    RSComplex _d_i = {0., 0.};
    RSComplex i = {0., 0.};
    RSComplex _d_one = {0., 0.};
    RSComplex one = {0., 0.};
    RSComplex _d_W = {0., 0.};
    RSComplex W = {0., 0.};
    RSComplex _d_sum = {0., 0.};
    RSComplex sum = {0., 0.};
    unsigned long _t0;
    int _d_n = 0;
    int n = 0;
    clad::tape<RSComplex> _t1 = {};
    RSComplex _d_t3 = {0., 0.};
    RSComplex t3 = {0., 0.};
    clad::tape<RSComplex> _t2 = {};
    RSComplex _d_top = {0., 0.};
    RSComplex top = {0., 0.};
    clad::tape<RSComplex> _t3 = {};
    RSComplex _d_t4 = {0., 0.};
    RSComplex t4 = {0., 0.};
    clad::tape<RSComplex> _t4 = {};
    RSComplex _d_t5 = {0., 0.};
    RSComplex t5 = {0., 0.};
    clad::tape<RSComplex> _t5 = {};
    RSComplex _d_bot = {0., 0.};
    RSComplex bot = {0., 0.};
    clad::tape<RSComplex> _t6 = {};
    RSComplex _d_t6 = {0., 0.};
    RSComplex t6 = {0., 0.};
    clad::tape<RSComplex> _t7 = {};
    clad::tape<clad::ValueAndAdjoint<RSComplex &, RSComplex &> > _t8 = {};
    RSComplex _t9;
    RSComplex _d_a = {0., 0.};
    RSComplex a = {0., 0.};
    RSComplex _d_b = {0., 0.};
    RSComplex b = {0., 0.};
    RSComplex _d_c = {0., 0.};
    RSComplex c = {0., 0.};
    RSComplex _d_d = {0., 0.};
    RSComplex d = {0., 0.};
    RSComplex _d_i0 = {0., 0.};
    RSComplex i0 = {0., 0.};
    RSComplex _d_Z2 = {0., 0.};
    RSComplex Z2 = {0., 0.};
    RSComplex _d_W0 = {0., 0.};
    RSComplex W0 = {0., 0.};
    {
        _cond0 = c_abs(Z) < 6.;
        if (_cond0) {
            prefactor = {0, 81.243300000000005};
            an = {0.27584019999999998, 0.224574, 0.1594149, 0.09866577, 0.053244140000000002, 0.025052149999999999, 0.01027747, 0.003676164, 0.0011464940000000001, 3.1175700000000002E-4};
            neg_1n = {-1., 1., -1., 1., -1., 1., -1., 1., -1., 1.};
            denominator_left = {9.8696040000000007, 39.47842, 88.826440000000005, 157.91370000000001, 246.74010000000001, 355.30579999999998, 483.61059999999998, 631.65470000000005, 799.43799999999999, 986.96040000000005};
            t1 = {0, 12};
            t2 = {12, 0};
            i = {0, 1};
            one = {1, 0};
            W = c_div(c_mul(i, c_sub(one, fast_cexp(c_mul(t1, Z)))), c_mul(t2, Z));
            sum = {0, 0};
            _t0 = 0UL;
            for (n = 0; ; n++) {
                {
                    if (!(n < 10))
                        break;
                }
                _t0++;
                // clad::push(_t1, std::move(t3));
                t3 = {neg_1n[n], 0};
                // clad::push(_t2, std::move(top));
                top = c_sub(c_mul(t3, fast_cexp(c_mul(t1, Z))), one);
                // clad::push(_t3, std::move(t4));
                t4 = {denominator_left[n], 0};
                // clad::push(_t4, std::move(t5));
                t5 = {144, 0};
                // clad::push(_t5, std::move(bot));
                bot = c_sub(t4, c_mul(t5, c_mul(Z, Z)));
                // clad::push(_t6, std::move(t6));
                t6 = {an[n], 0};
                // clad::push(_t7, sum);
                // clad::push(_t8,
				operator_equal_reverse_forw(sum, c_add(sum, c_mul(t6, c_div(top, bot))), &_d_sum, {0., 0.});
            }
            _t9 = W;
            clad::ValueAndAdjoint<RSComplex &, RSComplex &> _t10 = operator_equal_reverse_forw(W, c_add(W, c_mul(prefactor, c_mul(Z, sum))), &_d_W, {0., 0.});
            goto _label0;
        } else {
            a = {0.51242422475476845, 0};
            b = {0.27525512860841095, 0};
            c = {0.051765358792987826, 0};
            d = {2.7247448713915889, 0};
            i0 = {0, 1};
            Z2 = c_mul(Z, Z);
            W0 = c_mul(c_mul(Z, i0), c_add(c_div(a, c_sub(Z2, b)), c_div(c, c_sub(Z2, d))));
            goto _label1;
        }
    }
    if (_cond0) {
      _label0:
        clad::custom_derivatives::class_functions::constructor_pullback(std::move(W), &_d_y, &_d_W);
        {
            RSComplex _r31 = {0., 0.};
            W = _t9;
            operator_equal_pullback(W, c_add(W, c_mul(prefactor, c_mul(Z, sum))), {0., 0.}, &_d_W, &_r31);
            RSComplex _r32 = {0., 0.};
            RSComplex _r33 = {0., 0.};
            c_add_pullback(W, c_mul(prefactor, c_mul(Z, sum)), _r31, &_r32, &_r33);
            clad::custom_derivatives::class_functions::constructor_pullback(W, &_r32, &_d_W);
            RSComplex _r34 = {0., 0.};
            RSComplex _r35 = {0., 0.};
            c_mul_pullback(prefactor, c_mul(Z, sum), _r33, &_r34, &_r35);
            clad::custom_derivatives::class_functions::constructor_pullback(prefactor, &_r34, &_d_prefactor);
            RSComplex _r36 = {0., 0.};
            RSComplex _r37 = {0., 0.};
            c_mul_pullback(Z, sum, _r35, &_r36, &_r37);
            clad::custom_derivatives::class_functions::constructor_pullback(Z, &_r36, &(*_d_Z));
            clad::custom_derivatives::class_functions::constructor_pullback(sum, &_r37, &_d_sum);
        }
        for (;; _t0--) {
            {
                if (!_t0)
                    break;
            }
            n--;
            {
                RSComplex _r24 = {0., 0.};
                // sum = clad::back(_t7);
                operator_equal_pullback(sum, c_add(sum, c_mul(t6, c_div(top, bot))), {0., 0.}, &_d_sum, &_r24);
                RSComplex _r25 = {0., 0.};
                RSComplex _r26 = {0., 0.};
                c_add_pullback(sum, c_mul(t6, c_div(top, bot)), _r24, &_r25, &_r26);
                clad::custom_derivatives::class_functions::constructor_pullback(sum, &_r25, &_d_sum);
                RSComplex _r27 = {0., 0.};
                RSComplex _r28 = {0., 0.};
                c_mul_pullback(t6, c_div(top, bot), _r26, &_r27, &_r28);
                clad::custom_derivatives::class_functions::constructor_pullback(t6, &_r27, &_d_t6);
                RSComplex _r29 = {0., 0.};
                RSComplex _r30 = {0., 0.};
                c_div_pullback(top, bot, _r28, &_r29, &_r30);
                clad::custom_derivatives::class_functions::constructor_pullback(top, &_r29, &_d_top);
                clad::custom_derivatives::class_functions::constructor_pullback(bot, &_r30, &_d_bot);
                // clad::pop(_t7);
                // clad::pop(_t8);
            }
            {
                _d_an[n] += _d_t6.r;
                _d_t6 = {0., 0.};
                // t6 = clad::pop(_t6);
            }
            {
                RSComplex _r18 = {0., 0.};
                RSComplex _r19 = {0., 0.};
                c_sub_pullback(t4, c_mul(t5, c_mul(Z, Z)), _d_bot, &_r18, &_r19);
                clad::custom_derivatives::class_functions::constructor_pullback(t4, &_r18, &_d_t4);
                RSComplex _r20 = {0., 0.};
                RSComplex _r21 = {0., 0.};
                c_mul_pullback(t5, c_mul(Z, Z), _r19, &_r20, &_r21);
                clad::custom_derivatives::class_functions::constructor_pullback(t5, &_r20, &_d_t5);
                RSComplex _r22 = {0., 0.};
                RSComplex _r23 = {0., 0.};
                c_mul_pullback(Z, Z, _r21, &_r22, &_r23);
                clad::custom_derivatives::class_functions::constructor_pullback(Z, &_r22, &(*_d_Z));
                clad::custom_derivatives::class_functions::constructor_pullback(Z, &_r23, &(*_d_Z));
                _d_bot = {0., 0.};
                // bot = clad::pop(_t5);
            }
            {
                _d_t5 = {0., 0.};
                // t5 = clad::pop(_t4);
            }
            {
                _d_denominator_left[n] += _d_t4.r;
                _d_t4 = {0., 0.};
                // t4 = clad::pop(_t3);
            }
            {
                RSComplex _r11 = {0., 0.};
                RSComplex _r17 = {0., 0.};
                c_sub_pullback(c_mul(t3, fast_cexp(c_mul(t1, Z))), one, _d_top, &_r11, &_r17);
                RSComplex _r12 = {0., 0.};
                RSComplex _r13 = {0., 0.};
                c_mul_pullback(t3, fast_cexp(c_mul(t1, Z)), _r11, &_r12, &_r13);
                clad::custom_derivatives::class_functions::constructor_pullback(t3, &_r12, &_d_t3);
                RSComplex _r14 = {0., 0.};
                fast_cexp_pullback(c_mul(t1, Z), _r13, &_r14);
                RSComplex _r15 = {0., 0.};
                RSComplex _r16 = {0., 0.};
                c_mul_pullback(t1, Z, _r14, &_r15, &_r16);
                clad::custom_derivatives::class_functions::constructor_pullback(t1, &_r15, &_d_t1);
                clad::custom_derivatives::class_functions::constructor_pullback(Z, &_r16, &(*_d_Z));
                clad::custom_derivatives::class_functions::constructor_pullback(one, &_r17, &_d_one);
                _d_top = {0., 0.};
                // top = clad::pop(_t2);
            }
            {
                _d_neg_1n[n] += _d_t3.r;
                _d_t3 = {0., 0.};
                // t3 = clad::pop(_t1);
            }
        }
        {
            RSComplex _r0 = {0., 0.};
            RSComplex _r8 = {0., 0.};
            c_div_pullback(c_mul(i, c_sub(one, fast_cexp(c_mul(t1, Z)))), c_mul(t2, Z), _d_W, &_r0, &_r8);
            RSComplex _r1 = {0., 0.};
            RSComplex _r2 = {0., 0.};
            c_mul_pullback(i, c_sub(one, fast_cexp(c_mul(t1, Z))), _r0, &_r1, &_r2);
            clad::custom_derivatives::class_functions::constructor_pullback(
                i, &_r1, &_d_i);
            RSComplex _r3 = {0., 0.};
            RSComplex _r4 = {0., 0.};
            c_sub_pullback(one, fast_cexp(c_mul(t1, Z)), _r2, &_r3, &_r4);
            clad::custom_derivatives::class_functions::constructor_pullback(
                one, &_r3, &_d_one);
            RSComplex _r5 = {0., 0.};
            fast_cexp_pullback(c_mul(t1, Z), _r4, &_r5);
            RSComplex _r6 = {0., 0.};
            RSComplex _r7 = {0., 0.};
            c_mul_pullback(t1, Z, _r5, &_r6, &_r7);
            clad::custom_derivatives::class_functions::constructor_pullback(
                t1, &_r6, &_d_t1);
            clad::custom_derivatives::class_functions::constructor_pullback(
                Z, &_r7, &(*_d_Z));
            RSComplex _r9 = {0., 0.};
            RSComplex _r10 = {0., 0.};
            c_mul_pullback(t2, Z, _r8, &_r9, &_r10);
            clad::custom_derivatives::class_functions::constructor_pullback(
                t2, &_r9, &_d_t2);
            clad::custom_derivatives::class_functions::constructor_pullback(
                Z, &_r10, &(*_d_Z));
        }
    } else {
      _label1:
        clad::custom_derivatives::class_functions::constructor_pullback(std::move(W0), &_d_y, &_d_W0);
        {
            RSComplex _r40 = {0., 0.};
            RSComplex _r43 = {0., 0.};
            c_mul_pullback(c_mul(Z, i0), c_add(c_div(a, c_sub(Z2, b)), c_div(c, c_sub(Z2, d))), _d_W0, &_r40, &_r43);
            RSComplex _r41 = {0., 0.};
            RSComplex _r42 = {0., 0.};
            c_mul_pullback(Z, i0, _r40, &_r41, &_r42);
            clad::custom_derivatives::class_functions::constructor_pullback(Z, &_r41, &(*_d_Z));
            clad::custom_derivatives::class_functions::constructor_pullback(i0, &_r42, &_d_i0);
            RSComplex _r44 = {0., 0.};
            RSComplex _r49 = {0., 0.};
            c_add_pullback(c_div(a, c_sub(Z2, b)), c_div(c, c_sub(Z2, d)), _r43, &_r44, &_r49);
            RSComplex _r45 = {0., 0.};
            RSComplex _r46 = {0., 0.};
            c_div_pullback(a, c_sub(Z2, b), _r44, &_r45, &_r46);
            clad::custom_derivatives::class_functions::constructor_pullback(a, &_r45, &_d_a);
            RSComplex _r47 = {0., 0.};
            RSComplex _r48 = {0., 0.};
            c_sub_pullback(Z2, b, _r46, &_r47, &_r48);
            clad::custom_derivatives::class_functions::constructor_pullback(Z2, &_r47, &_d_Z2);
            clad::custom_derivatives::class_functions::constructor_pullback(b, &_r48, &_d_b);
            RSComplex _r50 = {0., 0.};
            RSComplex _r51 = {0., 0.};
            c_div_pullback(c, c_sub(Z2, d), _r49, &_r50, &_r51);
            clad::custom_derivatives::class_functions::constructor_pullback(c, &_r50, &_d_c);
            RSComplex _r52 = {0., 0.};
            RSComplex _r53 = {0., 0.};
            c_sub_pullback(Z2, d, _r51, &_r52, &_r53);
            clad::custom_derivatives::class_functions::constructor_pullback(Z2, &_r52, &_d_Z2);
            clad::custom_derivatives::class_functions::constructor_pullback(d, &_r53, &_d_d);
        }
        {
            RSComplex _r38 = {0., 0.};
            RSComplex _r39 = {0., 0.};
            c_mul_pullback(Z, Z, _d_Z2, &_r38, &_r39);
            clad::custom_derivatives::class_functions::constructor_pullback(Z, &_r38, &(*_d_Z));
            clad::custom_derivatives::class_functions::constructor_pullback(Z, &_r39, &(*_d_Z));
        }
    }
}
__attribute__((always_inline)) __attribute__((device)) void calculate_micro_xs_doppler_pullback(double *micro_xs, int nuc, double E, Input input, const int *n_windows, const double *pseudo_K0RS, const Window *windows, Pole *poles, int max_num_windows, int max_num_poles, double *_d_micro_xs, int *_d_nuc, double *_d_E, Pole *_d_poles, int *_d_max_num_windows, int *_d_max_num_poles) {
    bool _cond0;
    int _d_i = 0;
    int i = 0;
    clad::tape<Pole> _t1 = {};
    Pole _d_pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
    Pole pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
    clad::tape<RSComplex> _t2 = {};
    RSComplex _d_E_c = {0., 0.};
    RSComplex E_c = {0., 0.};
    clad::tape<RSComplex> _t3 = {};
    RSComplex _d_dopp_c = {0., 0.};
    RSComplex dopp_c = {0., 0.};
    clad::tape<RSComplex> _t4 = {};
    RSComplex _d_Z = {0., 0.};
    RSComplex Z = {0., 0.};
    clad::tape<RSComplex> _t5 = {};
    RSComplex _d_faddeeva = {0., 0.};
    RSComplex faddeeva = {0., 0.};
    clad::tape<RSComplex> _t6 = {};
    RSComplex _d__t1 = {0., 0.};
    RSComplex _t10 = {0., 0.};
    clad::tape<RSComplex> _t7 = {};
    RSComplex _d__t2 = {0., 0.};
    RSComplex _t20 = {0., 0.};
    clad::tape<RSComplex> _t8 = {};
    RSComplex _d__t3 = {0., 0.};
    RSComplex _t30 = {0., 0.};
    double _d_sigT = 0.;
    double sigT;
    double _d_sigA = 0.;
    double sigA;
    double _d_sigF = 0.;
    double sigF;
    double _d_sigE = 0.;
    double sigE;
    double _d_spacing = 0.;
    double spacing = 1. / n_windows[nuc];
    int _d_window = 0;
    int window = (int)(E / spacing);
    {
        _cond0 = window == n_windows[nuc];
        if (_cond0)
            window--;
    }
    RSComplex _d_sigTfactors[4] = {0};
    RSComplex sigTfactors[4];
    calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors);
    // printf("nuc = %d, max_num_poles = %d, i= %d\n", nuc, max_num_poles, i);
    Window _d_w = {0., 0., 0., 0, 0};
    Window w = windows[nuc * max_num_windows + window];
    sigT = E * w.T;
    sigA = E * w.A;
    sigF = E * w.F;
    double _d_dopp = 0.;
    double dopp = 0.5;
    unsigned long _t0 = 0UL;
    for (i = w.start; ; i++) {
        {
            if (!(i < w.end))
                break;
        }
        _t0++;
        // clad::push(_t1, std::move(pole));
        pole = poles[nuc * max_num_poles + i];
        // clad::push(_t2, std::move(E_c));
        E_c = {E, 0};
        // clad::push(_t3, std::move(dopp_c));
        dopp_c = {dopp, 0};
        // clad::push(_t4, std::move(Z));
        Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);
        // clad::push(_t5, std::move(faddeeva));
        faddeeva = fast_nuclear_W(Z);
        // clad::push(_t6, std::move(_t10));
        _t10 = c_mul(pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]));
        sigT += _t10.r;
        // clad::push(_t7, std::move(_t20));
        _t20 = c_mul(pole.MP_RA, faddeeva);
        sigA += _t20.r;
        // clad::push(_t8, std::move(_t30));
        _t30 = c_mul(pole.MP_RF, faddeeva);
        sigF += _t30.r;
    }
    sigE = sigT - sigA;
    micro_xs[0] = sigT;
    micro_xs[1] = sigA;
    micro_xs[2] = sigF;
    micro_xs[3] = sigE;
    {
        double _r_d10 = _d_micro_xs[3];
        _d_micro_xs[3] = 0.;
        _d_sigE += _r_d10;
    }
    {
        double _r_d9 = _d_micro_xs[2];
        _d_micro_xs[2] = 0.;
        _d_sigF += _r_d9;
    }
    {
        double _r_d8 = _d_micro_xs[1];
        _d_micro_xs[1] = 0.;
        _d_sigA += _r_d8;
    }
    {
        double _r_d7 = _d_micro_xs[0];
        _d_micro_xs[0] = 0.;
        _d_sigT += _r_d7;
    }
    {
        double _r_d6 = _d_sigE;
        _d_sigE = 0.;
        _d_sigT += _r_d6;
        _d_sigA += -_r_d6;
    }
    {
        for (;; _t0--) {
            {
                if (!_t0)
                    break;
            }
            i--;
            {
                double _r_d5 = _d_sigF;
                _d__t3.r += _r_d5;
            }
            {
                RSComplex _r15 = {0., 0.};
                RSComplex _r16 = {0., 0.};

                pole = poles[nuc * max_num_poles + i];
                Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);
                faddeeva = fast_nuclear_W(Z);

                c_mul_pullback(pole.MP_RF, faddeeva, _d__t3, &_r15, &_r16);
                clad::custom_derivatives::class_functions::constructor_pullback(pole.MP_RF, &_r15, &_d_pole.MP_RF);
                clad::custom_derivatives::class_functions::constructor_pullback(faddeeva, &_r16, &_d_faddeeva);
                _d__t3 = {0., 0.};
                // _t30 = clad::pop(_t8);
            }
            {
                double _r_d4 = _d_sigA;
                _d__t2.r += _r_d4;
            }
            {
                RSComplex _r13 = {0., 0.};
                RSComplex _r14 = {0., 0.};
                c_mul_pullback(pole.MP_RA, faddeeva, _d__t2, &_r13, &_r14);
                clad::custom_derivatives::class_functions::constructor_pullback(pole.MP_RA, &_r13, &_d_pole.MP_RA);
                clad::custom_derivatives::class_functions::constructor_pullback(faddeeva, &_r14, &_d_faddeeva);
                _d__t2 = {0., 0.};
                // _t20 = clad::pop(_t7);
            }
            {
                double _r_d3 = _d_sigT;
                _d__t1.r += _r_d3;
            }
            {
                RSComplex _r9 = {0., 0.};
                RSComplex _r10 = {0., 0.};
                c_mul_pullback(pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]), _d__t1, &_r9, &_r10);
                clad::custom_derivatives::class_functions::constructor_pullback(pole.MP_RT, &_r9, &_d_pole.MP_RT);
                RSComplex _r11 = {0., 0.};
                RSComplex _r12 = {0., 0.};
                c_mul_pullback(faddeeva, sigTfactors[pole.l_value], _r10, &_r11, &_r12);
                clad::custom_derivatives::class_functions::constructor_pullback(faddeeva, &_r11, &_d_faddeeva);
                clad::custom_derivatives::class_functions::constructor_pullback(sigTfactors[pole.l_value], &_r12, &_d_sigTfactors[pole.l_value]);
                _d__t1 = {0., 0.};
                // _t10 = clad::pop(_t6);
            }
            {
                RSComplex _r8 = {0., 0.};
                fast_nuclear_W_pullback(Z, _d_faddeeva, &_r8);
                clad::custom_derivatives::class_functions::constructor_pullback(Z, &_r8, &_d_Z);
                // printf("_d_Z = {%0.2f, %0.2f}\n", _d_Z.i, _d_Z.r);
                _d_faddeeva = {0., 0.};
                // faddeeva = clad::pop(_t5);
            }
            {
                RSComplex _r4 = {0., 0.};
                RSComplex _r7 = {0., 0.};
                c_mul_pullback(c_sub(E_c, pole.MP_EA), dopp_c, _d_Z, &_r4, &_r7);
                RSComplex _r5 = {0., 0.};
                RSComplex _r6 = {0., 0.};
                c_sub_pullback(E_c, pole.MP_EA, _r4, &_r5, &_r6);
                clad::custom_derivatives::class_functions::constructor_pullback(E_c, &_r5, &_d_E_c);
                clad::custom_derivatives::class_functions::constructor_pullback(pole.MP_EA, &_r6, &_d_pole.MP_EA);
                // printf("_d_pole.MP_EA = {%0.2f, %0.2f}\n", _d_pole.MP_EA.i,
                    //    _d_pole.MP_EA.r);
                clad::custom_derivatives::class_functions::constructor_pullback(dopp_c, &_r7, &_d_dopp_c);
                _d_Z = {0., 0.};
                // Z = clad::pop(_t4);
            }
            {
                _d_dopp += _d_dopp_c.r;
                _d_dopp_c = {0., 0.};
                // dopp_c = clad::pop(_t3);
            }
            {
                *_d_E += _d_E_c.r;
                _d_E_c = {0., 0.};
                // E_c = clad::pop(_t2);
            }
            {
                clad::custom_derivatives::class_functions::constructor_pullback(poles[nuc * max_num_poles + i], &_d_pole, &_d_poles[nuc * max_num_poles + i]);
                // printf("_d_poles[%d].MP_EA = {%0.2f, "
                //        "%0.2f}\n", nuc * max_num_poles + i,
                //        _d_poles[nuc * max_num_poles + i].MP_EA.i,
                //        _d_poles[nuc * max_num_poles + i].MP_EA.r);
                _d_pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
                // pole = clad::pop(_t1);
            }
        }
        _d_w.start += _d_i;
    }
    {
        double _r_d2 = _d_sigF;
        _d_sigF = 0.;
        *_d_E += _r_d2 * w.F;
        _d_w.F += E * _r_d2;
    }
    {
        double _r_d1 = _d_sigA;
        _d_sigA = 0.;
        *_d_E += _r_d1 * w.A;
        _d_w.A += E * _r_d1;
    }
    {
        double _r_d0 = _d_sigT;
        _d_sigT = 0.;
        *_d_E += _r_d0 * w.T;
        _d_w.T += E * _r_d0;
    }
    {
        *_d_E += _d_window / spacing;
        double _r1 = _d_window * -(E / (spacing * spacing));
        _d_spacing += _r1;
    }
    double _r0 = _d_spacing * -(1. / (n_windows[nuc] * n_windows[nuc]));
}
__attribute__((always_inline)) __attribute__((device)) void calculate_micro_xs_pullback(double *micro_xs, int nuc, double E, Input input, const int *n_windows, const double *pseudo_K0RS, const Window *windows, Pole *poles, int max_num_windows, int max_num_poles, double *_d_micro_xs, int *_d_nuc, double *_d_E, Pole *_d_poles, int *_d_max_num_windows, int *_d_max_num_poles) {
    bool _cond0;
    int _d_i = 0;
    int i = 0;
    clad::tape<RSComplex> _t1 = {};
    RSComplex _d_PSIIKI = {0., 0.};
    RSComplex PSIIKI = {0., 0.};
    clad::tape<RSComplex> _t2 = {};
    RSComplex _d_CDUM = {0., 0.};
    RSComplex CDUM = {0., 0.};
    clad::tape<Pole> _t3 = {};
    Pole _d_pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
    Pole pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
    clad::tape<RSComplex> _t4 = {};
    RSComplex _d_t1 = {0., 0.};
    RSComplex t1 = {0., 0.};
    clad::tape<RSComplex> _t5 = {};
    RSComplex _d_t2 = {0., 0.};
    RSComplex t2 = {0., 0.};
    clad::tape<RSComplex> _t6 = {};
    clad::tape<clad::ValueAndAdjoint<RSComplex &, RSComplex &> > _t7 = {};
    clad::tape<RSComplex> _t8 = {};
    RSComplex _d_E_c = {0., 0.};
    RSComplex E_c = {0., 0.};
    clad::tape<RSComplex> _t9 = {};
    clad::tape<clad::ValueAndAdjoint<RSComplex &, RSComplex &> > _t10 = {};
    clad::tape<RSComplex> _t12 = {};
    RSComplex _d__t1 = {0., 0.};
    RSComplex _t11 = {0., 0.};
    clad::tape<RSComplex> _t13 = {};
    RSComplex _d__t2 = {0., 0.};
    RSComplex _t20 = {0., 0.};
    clad::tape<RSComplex> _t14 = {};
    RSComplex _d__t3 = {0., 0.};
    RSComplex _t30 = {0., 0.};
    double _d_sigT = 0.;
    double sigT;
    double _d_sigA = 0.;
    double sigA;
    double _d_sigF = 0.;
    double sigF;
    double _d_sigE = 0.;
    double sigE;
    double _d_spacing = 0.;
    double spacing = 1. / n_windows[nuc];
    int _d_window = 0;
    int window = (int)(E / spacing);
    {
        _cond0 = window == n_windows[nuc];
        if (_cond0)
            window--;
    }
    RSComplex _d_sigTfactors[4] = {0};
    RSComplex sigTfactors[4];
    calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors);
    Window _d_w = {0., 0., 0., 0, 0};
    Window w = windows[nuc * max_num_windows + window];
    sigT = E * w.T;
    sigA = E * w.A;
    sigF = E * w.F;
    unsigned long _t0 = 0UL;
    for (i = w.start; ; i++) {
        {
            if (!(i < w.end))
                break;
        }
        _t0++;
        // clad::push(_t1, std::move(PSIIKI));
        PSIIKI = {0., 0.};
        // clad::push(_t2, std::move(CDUM));
       	CDUM = {0., 0.};
        // clad::push(_t3, std::move(pole));
        pole = poles[nuc * max_num_poles + i];
        // clad::push(_t4, std::move(t1));
        t1 = {0, 1};
        // clad::push(_t5, std::move(t2));
        t2 = {sqrt(E), 0};
        // clad::push(_t6, PSIIKI);
        // clad::push(_t7;
         operator_equal_reverse_forw(
                       PSIIKI, c_div(t1, c_sub(pole.MP_EA, t2)), &_d_PSIIKI,
                       {0., 0.});
        // clad::push(_t8, std::move(E_c));
        E_c = {E, 0};
        // clad::push(_t9, CDUM);
        // clad::push(_t10; 
        operator_equal_reverse_forw(CDUM, c_div(PSIIKI, E_c),
                                                     &_d_CDUM, {0., 0.});
        // clad::push(_t12, std::move(_t11));
        _t11 = c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value]));
        sigT += _t11.r;
        // clad::push(_t13, std::move(_t20));
        _t20 = c_mul(pole.MP_RA, CDUM);
        sigA += _t20.r;
        // clad::push(_t14, std::move(_t30));
        _t30 = c_mul(pole.MP_RF, CDUM);
        sigF += _t30.r;
    }
    sigE = sigT - sigA;
    micro_xs[0] = sigT;
    micro_xs[1] = sigA;
    micro_xs[2] = sigF;
    micro_xs[3] = sigE;
    {
        double _r_d10 = _d_micro_xs[3];
        _d_micro_xs[3] = 0.;
        _d_sigE += _r_d10;
    }
    {
        double _r_d9 = _d_micro_xs[2];
        _d_micro_xs[2] = 0.;
        _d_sigF += _r_d9;
    }
    {
        double _r_d8 = _d_micro_xs[1];
        _d_micro_xs[1] = 0.;
        _d_sigA += _r_d8;
    }
    {
        double _r_d7 = _d_micro_xs[0];
        _d_micro_xs[0] = 0.;
        _d_sigT += _r_d7;
    }
    {
        double _r_d6 = _d_sigE;
        _d_sigE = 0.;
        _d_sigT += _r_d6;
        _d_sigA += -_r_d6;
    }
    {
        for (;; _t0--) {
            {
                if (!_t0)
                    break;
            }
            i--;
            {
                double _r_d5 = _d_sigF;
                _d__t3.r += _r_d5;
            }
            {
                RSComplex _r19 = {0., 0.};
                RSComplex _r20 = {0., 0.};
                c_mul_pullback(pole.MP_RF, CDUM, _d__t3, &_r19, &_r20);
                clad::custom_derivatives::class_functions::constructor_pullback(pole.MP_RF, &_r19, &_d_pole.MP_RF);
                clad::custom_derivatives::class_functions::constructor_pullback(CDUM, &_r20, &_d_CDUM);
                _d__t3 = {0., 0.};
                // _t30 = clad::pop(_t14);
            }
            {
                double _r_d4 = _d_sigA;
                _d__t2.r += _r_d4;
            }
            {
                RSComplex _r17 = {0., 0.};
                RSComplex _r18 = {0., 0.};
                c_mul_pullback(pole.MP_RA, CDUM, _d__t2, &_r17, &_r18);
                clad::custom_derivatives::class_functions::constructor_pullback(pole.MP_RA, &_r17, &_d_pole.MP_RA);
                clad::custom_derivatives::class_functions::constructor_pullback(CDUM, &_r18, &_d_CDUM);
                _d__t2 = {0., 0.};
                // _t20 = clad::pop(_t13);
            }
            {
                double _r_d3 = _d_sigT;
                _d__t1.r += _r_d3;
            }
            {
                RSComplex _r13 = {0., 0.};
                RSComplex _r14 = {0., 0.};
                c_mul_pullback(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value]), _d__t1, &_r13, &_r14);
                clad::custom_derivatives::class_functions::constructor_pullback(pole.MP_RT, &_r13, &_d_pole.MP_RT);
                RSComplex _r15 = {0., 0.};
                RSComplex _r16 = {0., 0.};
                c_mul_pullback(CDUM, sigTfactors[pole.l_value], _r14, &_r15, &_r16);
                clad::custom_derivatives::class_functions::constructor_pullback(CDUM, &_r15, &_d_CDUM);
                clad::custom_derivatives::class_functions::constructor_pullback(sigTfactors[pole.l_value], &_r16, &_d_sigTfactors[pole.l_value]);
                _d__t1 = {0., 0.};
                // _t11 = clad::pop(_t12);
            }
            {
                RSComplex _r10 = {0., 0.};
                // CDUM = clad::back(_t9);
                operator_equal_pullback(CDUM, c_div(PSIIKI, E_c), {0., 0.}, &_d_CDUM, &_r10);
                RSComplex _r11 = {0., 0.};
                RSComplex _r12 = {0., 0.};
                c_div_pullback(PSIIKI, E_c, _r10, &_r11, &_r12);
                clad::custom_derivatives::class_functions::constructor_pullback(PSIIKI, &_r11, &_d_PSIIKI);
                clad::custom_derivatives::class_functions::constructor_pullback(E_c, &_r12, &_d_E_c);
                // clad::pop(_t9);
                // clad::pop(_t10);
            }
            {
                *_d_E += _d_E_c.r;
                _d_E_c = {0., 0.};
                // E_c = clad::pop(_t8);
            }
            {
                RSComplex _r5 = {0., 0.};
                // PSIIKI = clad::back(_t6);
                operator_equal_pullback(PSIIKI, c_div(t1, c_sub(pole.MP_EA, t2)), {0., 0.}, &_d_PSIIKI, &_r5);
                RSComplex _r6 = {0., 0.};
                RSComplex _r7 = {0., 0.};
                c_div_pullback(t1, c_sub(pole.MP_EA, t2), _r5, &_r6, &_r7);
                clad::custom_derivatives::class_functions::constructor_pullback(t1, &_r6, &_d_t1);
                RSComplex _r8 = {0., 0.};
                RSComplex _r9 = {0., 0.};
                c_sub_pullback(pole.MP_EA, t2, _r7, &_r8, &_r9);
                clad::custom_derivatives::class_functions::constructor_pullback(pole.MP_EA, &_r8, &_d_pole.MP_EA);
                clad::custom_derivatives::class_functions::constructor_pullback(t2, &_r9, &_d_t2);
                // clad::pop(_t6);
                // clad::pop(_t7);
            }
            {
                double _r4 = 0.;
                _r4 += _d_t2.r * clad::custom_derivatives::std::sqrt_pushforward(E, 1.).pushforward;
                *_d_E += _r4;
                _d_t2 = {0., 0.};
                // t2 = clad::pop(_t5);
            }
            {
                _d_t1 = {0., 0.};
                // t1 = clad::pop(_t4);
            }
            {
                clad::custom_derivatives::class_functions::constructor_pullback(poles[nuc * max_num_poles + i], &_d_pole, &_d_poles[nuc * max_num_poles + i]);
                _d_pole = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, 0};
                // pole = clad::pop(_t3);
            }
            {
                _d_CDUM = {0., 0.};
                // CDUM = clad::pop(_t2);
            }
            {
                _d_PSIIKI = {0., 0.};
                // PSIIKI = clad::pop(_t1);
            }
        }
        _d_w.start += _d_i;
    }
    {
        double _r_d2 = _d_sigF;
        _d_sigF = 0.;
        *_d_E += _r_d2 * w.F;
        _d_w.F += E * _r_d2;
    }
    {
        double _r_d1 = _d_sigA;
        _d_sigA = 0.;
        *_d_E += _r_d1 * w.A;
        _d_w.A += E * _r_d1;
    }
    {
        double _r_d0 = _d_sigT;
        _d_sigT = 0.;
        *_d_E += _r_d0 * w.T;
        _d_w.T += E * _r_d0;
    }
    {
        *_d_E += _d_window / spacing;
        double _r1 = _d_window * -(E / (spacing * spacing));
        _d_spacing += _r1;
    }
    double _r0 = _d_spacing * -(1. / (n_windows[nuc] * n_windows[nuc]));
}
__attribute__((device)) void calculate_macro_xs_grad_0_11(double *__restrict macro_xs, int mat, double E, Input input, const int *__restrict num_nucs, const int *__restrict mats, int max_num_nucs, const double *__restrict concs, const int *__restrict n_windows, const double *__restrict pseudo_K0Rs, const Window *__restrict windows, Pole *__restrict poles, int max_num_windows, int max_num_poles, double *_d_macro_xs, Pole *_d_poles) {
    int _d_mat = 0;
    double _d_E = 0.;
    Input _d_input = {0, 0, 0, static_cast<__hm>(0U), 0, 0, 0, 0, 0, 0, 0};
    int _d_max_num_nucs = 0;
    int _d_max_num_windows = 0;
    int _d_max_num_poles = 0;
    int _d_i = 0;
    int i = 0;
    double _d_micro_xs[4] = {0};
    clad::array<double> micro_xs(4UL);
    clad::tape<int> _t1 = {};
    int _d_nuc = 0;
    int nuc = 0;
    clad::tape<bool> _cond0 = {};
    clad::tape<unsigned long> _t2 = {};
    clad::tape<int> _t3 = {};
    int _d_j = 0;
    int j = 0;
    clad::tape<double> _t4 = {};
    int _d_sz = 0;
    int sz = num_nucs[mat];
    unsigned long _t0 = 0UL;
    for (i = 0; ; i++) {
        {
            if (!(i < sz))
                break;
        }
        _t0++;
        // clad::push(_t1, nuc);
        nuc = mats[mat * max_num_nucs + i];
        {
            clad::push(_cond0, input.doppler == 1);
            if (clad::back(_cond0))
                calculate_micro_xs_doppler(micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);
            else
                calculate_micro_xs(micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);
        }
        clad::push(_t2, 0UL);
        for (clad::push(_t3, j) , j = 0; ; j++) {
            {
                if (!(j < 4))
                    break;
            }
            clad::back(_t2)++;
            // clad::push(_t4, micro_xs[j]);
            macro_xs[j] += micro_xs[j] * concs[mat * max_num_nucs + i];
        }
    }
    for (;; _t0--) {
        {
            if (!_t0)
                break;
        }
        i--;
        nuc = mats[mat * max_num_nucs + i];
        {
            for (;; clad::back(_t2)--) {
                {
                    if (!clad::back(_t2))
                        break;
                }
                j--;
                {
                    double _r_d0 = _d_macro_xs[j];
                    _d_micro_xs[j] += _r_d0 * concs[mat * max_num_nucs + i];
                    // clad::pop(_t4);
                }
            }
            {
                _d_j = 0;
                j = clad::pop(_t3);
            }
            clad::pop(_t2);
        }
        {
            if (clad::back(_cond0)) {
                int _r0 = 0;
                double _r1 = 0.;
                int _r2 = 0;
                int _r3 = 0;
                calculate_micro_xs_doppler_pullback(micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles, _d_micro_xs, &_r0, &_r1, _d_poles, &_r2, &_r3);
                _d_nuc += _r0;
                _d_E += _r1;
                _d_max_num_windows += _r2;
                _d_max_num_poles += _r3;
            } else {
                int _r4 = 0;
                double _r5 = 0.;
                int _r6 = 0;
                int _r7 = 0;
                calculate_micro_xs_pullback(micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles, _d_micro_xs, &_r4, &_r5, _d_poles, &_r6, &_r7);
                _d_nuc += _r4;
                _d_E += _r5;
                _d_max_num_windows += _r6;
                _d_max_num_poles += _r7;
            }
            clad::pop(_cond0);
        }
        {
            _d_nuc = 0;
            // nuc = clad::pop(_t1);
        }
        clad::zero_init(_d_micro_xs);
    }
}