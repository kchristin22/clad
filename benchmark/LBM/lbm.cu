/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*############################################################################*/

// includes, system
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime_api.h>
// includes, project
#include "main.h"
#include "lbm.h"
#ifndef __MCUDA__
#include <cuda.h>
#else
#include <mcuda.h>
#endif

#define DFL1 (1.0f/ 3.0f)
#define DFL2 (1.0f/18.0f)
#define DFL3 (1.0f/36.0f)

// includes, kernels
#include "lbm_kernel.cu"

#include "clad/Differentiator/Differentiator.h"


/******************************************************************************/
void CUDA_LBM_performStreamCollide( float * srcGrid, float * dstGrid ) {
    non_differentiable dim3 dimBlock, dimGrid;
    dimBlock.x = SIZE_X;
	dimGrid.x = SIZE_Y;
	dimGrid.y = SIZE_Z;
	dimBlock.y = dimBlock.z = dimGrid.z = 1;
	performStreamCollide_kernel<<<dimGrid, dimBlock>>>(srcGrid, dstGrid);
//   CUDA_ERRCK;
}

void CUDA_LBM_kernel_loop_inner(int nTimeSteps, float srcGrid[SIZE],
		                          float dstGrid[SIZE]) {
    // int t;
    non_differentiable dim3 dimBlock, dimGrid;
    dimBlock.x = SIZE_X;
    dimGrid.x = SIZE_Y;
    dimGrid.y = SIZE_Z;
    dimBlock.y = dimBlock.z = dimGrid.z = 1;
    for (unsigned int i = 1; i <= nTimeSteps; i++)
    {
        // pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
        performStreamCollide_kernel<<<dimGrid, dimBlock>>>(srcGrid, dstGrid);
        LBM_Grid aux = srcGrid;
        srcGrid = dstGrid;
        dstGrid = aux;
        // pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
        // LBM_swapGrids(&CUDA_srcGrid, &CUDA_dstGrid);

/*
        if ((t & 63) == 0)
        {
            printf("timestep: %i\n", t);
#if 0
			CUDA_LBM_getDeviceGrid((float**)&CUDA_srcGrid, (float**)&TEMP_srcGrid);
			LBM_showGridStatistics( *TEMP_srcGrid );
#endif
        }
*/
    }
}

// void CUDA_LBM_performStreamCollide_grad(float *srcGrid, float *dstGrid, float *_d_srcGrid, float *_d_dstGrid);


// void CUDA_LBM_kernel_loop_inner_grad(int nTimeSteps, LBM_Grid srcGrid,
// 		                          LBM_Grid dstGrid, LBM_Grid _d_srcGrid, LBM_Grid _d_dstGrid) {
//     for (unsigned int i = 1; i <= nTimeSteps; i++)
//     {
//         CUDA_LBM_performStreamCollide_grad(srcGrid, dstGrid, _d_srcGrid, _d_dstGrid);
//         LBM_swapGrids(&srcGrid, &dstGrid);
//         LBM_swapGrids(&_d_srcGrid, &_d_dstGrid);
//     }
// }

void CUDA_LBM_kernel_loop(int nTimeSteps, LBM_Grid srcGrid,
                          LBM_Grid dstGrid, LBM_Grid srcGridb, LBM_Grid dstGridb) {
	constexpr size_t size   = TOTAL_PADDED_CELLS*N_CELL_ENTRIES*sizeof( float ) + 2*TOTAL_MARGIN*sizeof( float );
	constexpr size_t start = 15489;// + REAL_MARGIN;
#ifdef ALLOW_AD
#ifdef VERIFY
    cudaMemset(srcGridb - REAL_MARGIN, 0, size);
    cudaMemset(dstGridb - REAL_MARGIN, 0, size);

    float *here = new float[N];
    memset(here, 0, N * sizeof(float));
    here[0] = 1.0;
    cudaMemcpy(srcGridb + start, &here[0], N * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(&here[0], srcGrid + start, N * sizeof(float),
               cudaMemcpyDeviceToHost);
#endif
    auto grad = clad::gradient(CUDA_LBM_kernel_loop_inner,
       "srcGrid, dstGrid");
    // grad.execute(nTimeSteps, srcGrid, dstGrid, srcGridb, dstGridb);
    // CUDA_LBM_kernel_loop_inner_grad(nTimeSteps, srcGrid, dstGrid, srcGridb,
                                        // dstGridb);

#ifdef VERIFY
    cudaMemcpy(&here[0], srcGridb + start, N * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        printf("out here[%d]=%f\n", i, here[i]);
    printf("der=%f\n", here[0]);
#endif
#else
#ifdef VERIFY

    float *cache = new float[size / sizeof(float)];

    cudaMemcpy(&cache[0], srcGrid - REAL_MARGIN, size, cudaMemcpyDeviceToHost);
#endif

    // CUDA_LBM_kernel_loop_inner(nTimeSteps, srcGrid, dstGrid);

#ifdef VERIFY
    constexpr size_t N = 1;
#define PREC 1e-2
    float *here = new float[N];
    cudaMemcpy(&here[0], srcGrid + start, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    cache[start + REAL_MARGIN] += PREC;
    cudaMemcpy(srcGrid - REAL_MARGIN, &cache[0], size, cudaMemcpyHostToDevice);

    CUDA_LBM_kernel_loop_inner(nTimeSteps, srcGrid, dstGrid);

    float *here2 = new float[N];
    cudaMemcpy(&here2[0], srcGrid + start, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        printf("real PREC=%e here[%d]=%f here2=%f dif=%e, der=%f\n", PREC, i,
               here[i], here2[i], here2[i] - here[i],
               (here2[i] - here[i]) / PREC);
#endif

#endif
}

/*############################################################################*/

void LBM_allocateGrid( float** ptr ) {
	const size_t size   = TOTAL_PADDED_CELLS*N_CELL_ENTRIES*sizeof( float ) + 2*TOTAL_MARGIN*sizeof( float );

	*ptr = (float*)malloc( size );
	if( ! *ptr ) {
		printf( "LBM_allocateGrid: could not allocate %.1f MByte\n",
				size / (1024.0*1024.0) );
		exit( 1 );
	}

	memset( *ptr, 0, size );

	printf( "LBM_allocateGrid: allocated %.1f MByte\n",
			size / (1024.0*1024.0) );
	*ptr += REAL_MARGIN;
}

/******************************************************************************/

void CUDA_LBM_allocateGrid( float** ptr ) {
	const size_t size = TOTAL_PADDED_CELLS*N_CELL_ENTRIES*sizeof( float ) + 2*TOTAL_MARGIN*sizeof( float );
	cudaMalloc((void**)ptr, size);
        CUDA_ERRCK;
	*ptr += REAL_MARGIN;
}

/*############################################################################*/

void LBM_freeGrid( float** ptr ) {
	free( *ptr-REAL_MARGIN );
	*ptr = NULL;
}

/******************************************************************************/

void CUDA_LBM_freeGrid( float** ptr ) {
	cudaFree( *ptr-REAL_MARGIN );
	*ptr = NULL;
}

/*############################################################################*/

void LBM_initializeGrid( LBM_Grid grid ) {
	SWEEP_VAR

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
	SRC_C( grid  ) = DFL1;
	SRC_N( grid  ) = DFL2;
	SRC_S( grid  ) = DFL2;
	SRC_E( grid  ) = DFL2;
	SRC_W( grid  ) = DFL2;
	SRC_T( grid  ) = DFL2;
	SRC_B( grid  ) = DFL2;
	SRC_NE( grid ) = DFL3;
	SRC_NW( grid ) = DFL3;
	SRC_SE( grid ) = DFL3;
	SRC_SW( grid ) = DFL3;
	SRC_NT( grid ) = DFL3;
	SRC_NB( grid ) = DFL3;
	SRC_ST( grid ) = DFL3;
	SRC_SB( grid ) = DFL3;
	SRC_ET( grid ) = DFL3;
	SRC_EB( grid ) = DFL3;
	SRC_WT( grid ) = DFL3;
	SRC_WB( grid ) = DFL3;

	CLEAR_ALL_FLAGS_SWEEP( grid );
	SWEEP_END
}

/******************************************************************************/

void CUDA_LBM_initializeGrid( float** d_grid, float** h_grid ) {
	const size_t size   = TOTAL_PADDED_CELLS*N_CELL_ENTRIES*sizeof( float ) + 2*TOTAL_MARGIN*sizeof( float );

	cudaMemcpy(*d_grid - REAL_MARGIN, *h_grid - REAL_MARGIN, size, cudaMemcpyHostToDevice);
        CUDA_ERRCK;
}

void CUDA_LBM_getDeviceGrid( float** d_grid, float** h_grid ) {
	const size_t size   = TOTAL_PADDED_CELLS*N_CELL_ENTRIES*sizeof( float ) + 2*TOTAL_MARGIN*sizeof( float );
        cudaThreadSynchronize();
        CUDA_ERRCK;
	cudaMemcpy(*h_grid - REAL_MARGIN, *d_grid - REAL_MARGIN, size, cudaMemcpyDeviceToHost);
        CUDA_ERRCK;
}

/*############################################################################*/

void LBM_swapGrids( LBM_GridPtr grid1, LBM_GridPtr grid2 ) {
	LBM_Grid aux = *grid1;
	*grid1 = *grid2;
	*grid2 = aux;
}

/*############################################################################*/

void LBM_loadObstacleFile( LBM_Grid grid, const char* filename ) {
	int x,  y,  z;

	FILE* file = fopen( filename, "rb" );

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( fgetc( file ) != '.' ) SET_FLAG( grid, x, y, z, OBSTACLE );
			}
			fgetc( file );
		}
		fgetc( file );
	}

	fclose( file );
}

/*############################################################################*/

void LBM_initializeSpecialCellsForLDC( LBM_Grid grid ) {
	int x,  y,  z;

	for( z = -2; z < SIZE_Z+2; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( x == 0 || x == SIZE_X-1 ||
						y == 0 || y == SIZE_Y-1 ||
						z == 0 || z == SIZE_Z-1 ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
				else {
					if( (z == 1 || z == SIZE_Z-2) &&
							x > 1 && x < SIZE_X-2 &&
							y > 1 && y < SIZE_Y-2 ) {
						SET_FLAG( grid, x, y, z, ACCEL );
					}
				}
			}
		}
	}
}

/*############################################################################*/

void LBM_showGridStatistics( LBM_Grid grid ) {
	int nObstacleCells = 0,
	    nAccelCells    = 0,
	    nFluidCells    = 0;
	float ux, uy, uz;
	float minU2  = 1e+30, maxU2  = -1e+30, u2;
	float minRho = 1e+30, maxRho = -1e+30, rho;
	float mass = 0;

	SWEEP_VAR

		SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		rho = LOCAL( grid, C  ) + LOCAL( grid, N  )
		+ LOCAL( grid, S  ) + LOCAL( grid, E  )
		+ LOCAL( grid, W  ) + LOCAL( grid, T  )
		+ LOCAL( grid, B  ) + LOCAL( grid, NE )
		+ LOCAL( grid, NW ) + LOCAL( grid, SE )
		+ LOCAL( grid, SW ) + LOCAL( grid, NT )
		+ LOCAL( grid, NB ) + LOCAL( grid, ST )
		+ LOCAL( grid, SB ) + LOCAL( grid, ET )
		+ LOCAL( grid, EB ) + LOCAL( grid, WT )
		+ LOCAL( grid, WB );
	if( rho < minRho ) minRho = rho;
	if( rho > maxRho ) maxRho = rho;
	mass += rho;

	if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
		nObstacleCells++;
	}
	else {
		if( TEST_FLAG_SWEEP( grid, ACCEL ))
			nAccelCells++;
		else
			nFluidCells++;

		ux = + LOCAL( grid, E  ) - LOCAL( grid, W  )
			+ LOCAL( grid, NE ) - LOCAL( grid, NW )
			+ LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, ET ) + LOCAL( grid, EB )
			- LOCAL( grid, WT ) - LOCAL( grid, WB );
		uy = + LOCAL( grid, N  ) - LOCAL( grid, S  )
			+ LOCAL( grid, NE ) + LOCAL( grid, NW )
			- LOCAL( grid, SE ) - LOCAL( grid, SW )
			+ LOCAL( grid, NT ) + LOCAL( grid, NB )
			- LOCAL( grid, ST ) - LOCAL( grid, SB );
		uz = + LOCAL( grid, T  ) - LOCAL( grid, B  )
			+ LOCAL( grid, NT ) - LOCAL( grid, NB )
			+ LOCAL( grid, ST ) - LOCAL( grid, SB )
			+ LOCAL( grid, ET ) - LOCAL( grid, EB )
			+ LOCAL( grid, WT ) - LOCAL( grid, WB );
		u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
		if( u2 < minU2 ) minU2 = u2;
		if( u2 > maxU2 ) maxU2 = u2;
	}
	SWEEP_END

		printf( "LBM_showGridStatistics:\n"
				"\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
				"\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
				"\tminU: %e maxU: %e\n\n",
				nObstacleCells, nAccelCells, nFluidCells,
				minRho, maxRho, mass,
				sqrt( minU2 ), sqrt( maxU2 ) );

}

/*############################################################################*/

static void storeValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		const char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		int i;

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			buffer[i] = vPtr[sizeof( OUTPUT_PRECISION ) - i - 1];

		fwrite( buffer, sizeof( OUTPUT_PRECISION ), 1, file );
	}
	else {                                                     /* little endian */
		fwrite( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

/*############################################################################*/

void LBM_storeVelocityField( LBM_Grid grid, const char* filename,
		const int binary ) {
	OUTPUT_PRECISION rho, ux, uy, uz;

	FILE* file = fopen( filename, (binary ? "wb" : "w") );

	SWEEP_VAR
	SWEEP_START(0,0,0,SIZE_X,SIZE_Y,SIZE_Z)
				rho = + SRC_C( grid ) + SRC_N( grid )
					+ SRC_S( grid ) + SRC_E( grid )
					+ SRC_W( grid ) + SRC_T( grid )
					+ SRC_B( grid ) + SRC_NE( grid )
					+ SRC_NW( grid ) + SRC_SE( grid )
					+ SRC_SW( grid ) + SRC_NT( grid )
					+ SRC_NB( grid ) + SRC_ST( grid )
					+ SRC_SB( grid ) + SRC_ET( grid )
					+ SRC_EB( grid ) + SRC_WT( grid )
					+ SRC_WB( grid );
				ux = + SRC_E( grid ) - SRC_W( grid ) 
					+ SRC_NE( grid ) - SRC_NW( grid ) 
					+ SRC_SE( grid ) - SRC_SW( grid ) 
					+ SRC_ET( grid ) + SRC_EB( grid ) 
					- SRC_WT( grid ) - SRC_WB( grid );
				uy = + SRC_N( grid ) - SRC_S( grid ) 
					+ SRC_NE( grid ) + SRC_NW( grid ) 
					- SRC_SE( grid ) - SRC_SW( grid ) 
					+ SRC_NT( grid ) + SRC_NB( grid ) 
					- SRC_ST( grid ) - SRC_SB( grid );
				uz = + SRC_T( grid ) - SRC_B( grid ) 
					+ SRC_NT( grid ) - SRC_NB( grid ) 
					+ SRC_ST( grid ) - SRC_SB( grid ) 
					+ SRC_ET( grid ) - SRC_EB( grid ) 
					+ SRC_WT( grid ) - SRC_WB( grid );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					/*
					   fwrite( &ux, sizeof( ux ), 1, file );
					   fwrite( &uy, sizeof( uy ), 1, file );
					   fwrite( &uz, sizeof( uz ), 1, file );
					   */
					storeValue( file, &ux );
					storeValue( file, &uy );
					storeValue( file, &uz );
				} else
					fprintf( file, "%e %e %e\n", ux, uy, uz );

	SWEEP_END;

	fclose( file );
}

// __attribute__((global)) void performStreamCollide_kernel_pullback(float *srcGrid, float *dstGrid, float *_d_srcGrid, float *_d_dstGrid) {
//     bool _cond0;
//     float _d_ux = 0.F, _d_uy = 0.F, _d_uz = 0.F, _d_rho = 0.F, _d_u2 = 0.F;
//     float ux;
//     float uy;
//     float uz;
//     float rho;
//     float u2;
//     float _d_temp1 = 0.F, _d_temp2 = 0.F, _d_temp_base = 0.F;
//     float temp1;
//     float temp2;
//     float temp_base;
//     float _t0;
//     float _t1;
//     float _t2;
//     bool _cond1;
//     float _t3;
//     float _t4;
//     float _t5;
//     float _t6;
//     float _t7;
//     float _t8;
//     float _t9;
//     float _t10;
//     float _t11;
//     float _t12;
//     float _t13;
//     float _t14;
//     float _t15;
//     float _t16;
//     float _t17;
//     float _t18;
//     float _t19;
//     float _t20;
//     float _t21;
//     float _t22;
//     float _t23;
//     int __temp_x__0, __temp_y__0, __temp_z__0;
//     __temp_x__0 = threadIdx.x;
//     __temp_y__0 = blockIdx.x;
//     __temp_z__0 = blockIdx.y;
//     float _d_temp_swp = 0.F, _d_tempC = 0.F, _d_tempN = 0.F, _d_tempS = 0.F, _d_tempE = 0.F, _d_tempW = 0.F, _d_tempT = 0.F, _d_tempB = 0.F;
//     float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
//     float _d_tempNE = 0.F, _d_tempNW = 0.F, _d_tempSE = 0.F, _d_tempSW = 0.F, _d_tempNT = 0.F, _d_tempNB = 0.F, _d_tempST = 0.F;
//     float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST;
//     float _d_tempSB = 0.F, _d_tempET = 0.F, _d_tempEB = 0.F, _d_tempWT = 0.F, _d_tempWB = 0.F;
//     float tempSB, tempET, tempEB, tempWT, tempWB;
//     tempC = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempN = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * N + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempS = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * S + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempE = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * E + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempW = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * W + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempT = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * T + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempB = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * B + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempNE = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NE + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempNW = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NW + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempSE = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SE + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempSW = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SW + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempNT = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NT + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempNB = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NB + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempST = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ST + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempSB = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SB + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempET = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ET + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempEB = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * EB + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempWT = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WT + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     tempWB = srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WB + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     {
//         _cond0 = (*(unsigned int *)(void *)&srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * FLAGS + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] & OBSTACLE);
//         if (_cond0) {
//             temp_swp = tempN;
//             tempN = tempS;
//             tempS = temp_swp;
//             temp_swp = tempE;
//             tempE = tempW;
//             tempW = temp_swp;
//             temp_swp = tempT;
//             tempT = tempB;
//             tempB = temp_swp;
//             temp_swp = tempNE;
//             tempNE = tempSW;
//             tempSW = temp_swp;
//             temp_swp = tempNW;
//             tempNW = tempSE;
//             tempSE = temp_swp;
//             temp_swp = tempNT;
//             tempNT = tempSB;
//             tempSB = temp_swp;
//             temp_swp = tempNB;
//             tempNB = tempST;
//             tempST = temp_swp;
//             temp_swp = tempET;
//             tempET = tempWB;
//             tempWB = temp_swp;
//             temp_swp = tempEB;
//             tempEB = tempWT;
//             tempWT = temp_swp;
//         } else {
//             rho = tempC + tempN + tempS + tempE + tempW + tempT + tempB + tempNE + tempNW + tempSE + tempSW + tempNT + tempNB + tempST + tempSB + tempET + tempEB + tempWT + tempWB;
//             ux = +tempE - tempW + tempNE - tempNW + tempSE - tempSW + tempET + tempEB - tempWT - tempWB;
//             uy = +tempN - tempS + tempNE + tempNW - tempSE - tempSW + tempNT + tempNB - tempST - tempSB;
//             uz = +tempT - tempB + tempNT - tempNB + tempST - tempSB + tempET - tempEB + tempWT - tempWB;
//             _t0 = ux;
//             ux /= rho;
//             _t1 = uy;
//             uy /= rho;
//             _t2 = uz;
//             uz /= rho;
//             {
//                 _cond1 = (*(unsigned int *)(void *)&srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * FLAGS + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] & ACCEL);
//                 if (_cond1) {
//                     ux = 0.005F;
//                     uy = 0.002F;
//                     uz = 0.F;
//                 }
//             }
//             u2 = 1.5F * (ux * ux + uy * uy + uz * uz) - 1.F;
//             temp_base = 1.95F * rho;
//             temp1 = (1.F / 3.F) * temp_base;
//             temp_base = 1.95000005F * rho;
//             temp1 = (1.F / 3.F) * temp_base;
//             temp2 = 1.F - 1.95F;
//             _t3 = tempC;
//             tempC = temp2 * tempC + temp1 * -u2;
//             _t4 = temp1;
//             temp1 = (1.F / 18.F) * temp_base;
//             _t5 = tempN;
//             tempN = temp2 * tempN + temp1 * (uy * (4.5F * uy + 3.F) - u2);
//             _t6 = tempS;
//             tempS = temp2 * tempS + temp1 * (uy * (4.5F * uy - 3.F) - u2);
//             _t7 = tempT;
//             tempT = temp2 * tempT + temp1 * (uz * (4.5F * uz + 3.F) - u2);
//             _t8 = tempB;
//             tempB = temp2 * tempB + temp1 * (uz * (4.5F * uz - 3.F) - u2);
//             _t9 = tempE;
//             tempE = temp2 * tempE + temp1 * (ux * (4.5F * ux + 3.F) - u2);
//             _t10 = tempW;
//             tempW = temp2 * tempW + temp1 * (ux * (4.5F * ux - 3.F) - u2);
//             _t11 = temp1;
//             temp1 = (1.F / 36.F) * temp_base;
//             _t12 = tempNT;
//             tempNT = temp2 * tempNT + temp1 * ((+uy + uz) * (4.5F * (+uy + uz) + 3.F) - u2);
//             _t13 = tempNB;
//             tempNB = temp2 * tempNB + temp1 * ((+uy - uz) * (4.5F * (+uy - uz) + 3.F) - u2);
//             _t14 = tempST;
//             tempST = temp2 * tempST + temp1 * ((-uy + uz) * (4.5F * (-uy + uz) + 3.F) - u2);
//             _t15 = tempSB;
//             tempSB = temp2 * tempSB + temp1 * ((-uy - uz) * (4.5F * (-uy - uz) + 3.F) - u2);
//             _t16 = tempNE;
//             tempNE = temp2 * tempNE + temp1 * ((+ux + uy) * (4.5F * (+ux + uy) + 3.F) - u2);
//             _t17 = tempSE;
//             tempSE = temp2 * tempSE + temp1 * ((+ux - uy) * (4.5F * (+ux - uy) + 3.F) - u2);
//             _t18 = tempET;
//             tempET = temp2 * tempET + temp1 * ((+ux + uz) * (4.5F * (+ux + uz) + 3.F) - u2);
//             _t19 = tempEB;
//             tempEB = temp2 * tempEB + temp1 * ((+ux - uz) * (4.5F * (+ux - uz) + 3.F) - u2);
//             _t20 = tempNW;
//             tempNW = temp2 * tempNW + temp1 * ((-ux + uy) * (4.5F * (-ux + uy) + 3.F) - u2);
//             _t21 = tempSW;
//             tempSW = temp2 * tempSW + temp1 * ((-ux - uy) * (4.5F * (-ux - uy) + 3.F) - u2);
//             _t22 = tempWT;
//             tempWT = temp2 * tempWT + temp1 * ((-ux + uz) * (4.5F * (-ux + uz) + 3.F) - u2);
//             _t23 = tempWB;
//             tempWB = temp2 * tempWB + temp1 * ((-ux - uz) * (4.5F * (-ux - uz) + 3.F) - u2);
//         }
//     }
//     if ((((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0))) == 15489)
//         printf("C: (%d, %d, %d)\n", __temp_x__0, __temp_y__0, __temp_z__0);
//     float _t24 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempC;
//     if ((((120 + 8) * (120 + 0) * (150 + 4)) * N + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0))) == 15489)
//         printf("N: (%d, %d, %d)\n", __temp_x__0, __temp_y__0, __temp_z__0);
//     float _t25 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * N + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * N + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempN;
//     if ((((120 + 8) * (120 + 0) * (150 + 4)) * S + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0))) == 15489)
//         printf("S: (%d, %d, %d)\n", __temp_x__0, __temp_y__0, __temp_z__0);
//     float _t26 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * S + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * S + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempS;
//     float _t27 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * E + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * E + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempE;
//     float _t28 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * W + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * W + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempW;
//     float _t29 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * T + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * T + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempT;
//     float _t30 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * B + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * B + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempB;
//     float _t31 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NE + ((+1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NE + ((+1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempNE;
//     float _t32 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NW + ((-1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NW + ((-1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempNW;
//     float _t33 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SE + ((+1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SE + ((+1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempSE;
//     float _t34 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SW + ((-1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SW + ((-1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempSW;
//     float _t35 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NT + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NT + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempNT;
//     float _t36 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NB + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NB + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempNB;
//     float _t37 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ST + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ST + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempST;
//     float _t38 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SB + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SB + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempSB;
//     float _t39 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ET + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ET + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempET;
//     float _t40 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * EB + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * EB + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempEB;
//     float _t41 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WT + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WT + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempWT;
//     float _t42 = dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WB + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//     dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WB + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = tempWB;
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WB + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t42;
//         float _r_d101 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WB + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WB + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempWB += _r_d101;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WT + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t41;
//         float _r_d100 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WT + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WT + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempWT += _r_d100;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * EB + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t40;
//         float _r_d99 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * EB + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * EB + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempEB += _r_d99;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ET + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t39;
//         float _r_d98 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ET + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ET + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempET += _r_d98;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SB + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t38;
//         float _r_d97 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SB + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SB + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempSB += _r_d97;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ST + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t37;
//         float _r_d96 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ST + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ST + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempST += _r_d96;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NB + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t36;
//         float _r_d95 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NB + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NB + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempNB += _r_d95;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NT + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t35;
//         float _r_d94 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NT + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NT + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempNT += _r_d94;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SW + ((-1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t34;
//         float _r_d93 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SW + ((-1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SW + ((-1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempSW += _r_d93;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SE + ((+1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t33;
//         float _r_d92 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SE + ((+1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SE + ((+1 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempSE += _r_d92;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NW + ((-1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t32;
//         float _r_d91 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NW + ((-1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NW + ((-1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempNW += _r_d91;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NE + ((+1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t31;
//         float _r_d90 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NE + ((+1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NE + ((+1 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempNE += _r_d90;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * B + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t30;
//         float _r_d89 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * B + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * B + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (-1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempB += _r_d89;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * T + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t29;
//         float _r_d88 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * T + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * T + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (+1 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempT += _r_d88;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * W + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t28;
//         float _r_d87 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * W + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * W + ((-1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempW += _r_d87;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * E + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t27;
//         float _r_d86 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * E + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * E + ((+1 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempE += _r_d86;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * S + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t26;
//         float _r_d85 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * S + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * S + ((0 + __temp_x__0) + (-1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempS += _r_d85;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * N + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t25;
//         float _r_d84 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * N + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * N + ((0 + __temp_x__0) + (+1 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempN += _r_d84;
//     }
//     {
//         dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = _t24;
//         float _r_d83 = _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))];
//         _d_dstGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] = 0.F;
//         _d_tempC += _r_d83;
//         if ((((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0))) == 15489)
//             printf("_d_dstGrid: %f\n", _r_d83);
//     }
//     if (_cond0) {
//         {
//             float _r_d45 = _d_tempWT;
//             _d_tempWT = 0.F;
//             _d_temp_swp += _r_d45;
//         }
//         {
//             float _r_d44 = _d_tempEB;
//             _d_tempEB = 0.F;
//             _d_tempWT += _r_d44;
//         }
//         {
//             float _r_d43 = _d_temp_swp;
//             _d_temp_swp = 0.F;
//             _d_tempEB += _r_d43;
//         }
//         {
//             float _r_d42 = _d_tempWB;
//             _d_tempWB = 0.F;
//             _d_temp_swp += _r_d42;
//         }
//         {
//             float _r_d41 = _d_tempET;
//             _d_tempET = 0.F;
//             _d_tempWB += _r_d41;
//         }
//         {
//             float _r_d40 = _d_temp_swp;
//             _d_temp_swp = 0.F;
//             _d_tempET += _r_d40;
//         }
//         {
//             float _r_d39 = _d_tempST;
//             _d_tempST = 0.F;
//             _d_temp_swp += _r_d39;
//         }
//         {
//             float _r_d38 = _d_tempNB;
//             _d_tempNB = 0.F;
//             _d_tempST += _r_d38;
//         }
//         {
//             float _r_d37 = _d_temp_swp;
//             _d_temp_swp = 0.F;
//             _d_tempNB += _r_d37;
//         }
//         {
//             float _r_d36 = _d_tempSB;
//             _d_tempSB = 0.F;
//             _d_temp_swp += _r_d36;
//         }
//         {
//             float _r_d35 = _d_tempNT;
//             _d_tempNT = 0.F;
//             _d_tempSB += _r_d35;
//         }
//         {
//             float _r_d34 = _d_temp_swp;
//             _d_temp_swp = 0.F;
//             _d_tempNT += _r_d34;
//         }
//         {
//             float _r_d33 = _d_tempSE;
//             _d_tempSE = 0.F;
//             _d_temp_swp += _r_d33;
//         }
//         {
//             float _r_d32 = _d_tempNW;
//             _d_tempNW = 0.F;
//             _d_tempSE += _r_d32;
//         }
//         {
//             float _r_d31 = _d_temp_swp;
//             _d_temp_swp = 0.F;
//             _d_tempNW += _r_d31;
//         }
//         {
//             float _r_d30 = _d_tempSW;
//             _d_tempSW = 0.F;
//             _d_temp_swp += _r_d30;
//         }
//         {
//             float _r_d29 = _d_tempNE;
//             _d_tempNE = 0.F;
//             _d_tempSW += _r_d29;
//         }
//         {
//             float _r_d28 = _d_temp_swp;
//             _d_temp_swp = 0.F;
//             _d_tempNE += _r_d28;
//         }
//         {
//             float _r_d27 = _d_tempB;
//             _d_tempB = 0.F;
//             _d_temp_swp += _r_d27;
//         }
//         {
//             float _r_d26 = _d_tempT;
//             _d_tempT = 0.F;
//             _d_tempB += _r_d26;
//         }
//         {
//             float _r_d25 = _d_temp_swp;
//             _d_temp_swp = 0.F;
//             _d_tempT += _r_d25;
//         }
//         {
//             float _r_d24 = _d_tempW;
//             _d_tempW = 0.F;
//             _d_temp_swp += _r_d24;
//         }
//         {
//             float _r_d23 = _d_tempE;
//             _d_tempE = 0.F;
//             _d_tempW += _r_d23;
//         }
//         {
//             float _r_d22 = _d_temp_swp;
//             _d_temp_swp = 0.F;
//             _d_tempE += _r_d22;
//         }
//         {
//             float _r_d21 = _d_tempS;
//             _d_tempS = 0.F;
//             _d_temp_swp += _r_d21;
//         }
//         {
//             float _r_d20 = _d_tempN;
//             _d_tempN = 0.F;
//             _d_tempS += _r_d20;
//         }
//         {
//             float _r_d19 = _d_temp_swp;
//             _d_temp_swp = 0.F;
//             _d_tempN += _r_d19;
//         }
//     } else {
//         {
//             tempWB = _t23;
//             float _r_d82 = _d_tempWB;
//             _d_tempWB = 0.F;
//             _d_temp2 += _r_d82 * tempWB;
//             _d_tempWB += temp2 * _r_d82;
//             _d_temp1 += _r_d82 * ((-ux - uz) * (4.5F * (-ux - uz) + 3.F) - u2);
//             _d_ux += -temp1 * _r_d82 * (4.5F * (-ux - uz) + 3.F);
//             _d_uz += -temp1 * _r_d82 * (4.5F * (-ux - uz) + 3.F);
//             _d_ux += -4.5F * (-ux - uz) * temp1 * _r_d82;
//             _d_uz += -4.5F * (-ux - uz) * temp1 * _r_d82;
//             _d_u2 += -temp1 * _r_d82;
//         }
//         {
//             tempWT = _t22;
//             float _r_d81 = _d_tempWT;
//             _d_tempWT = 0.F;
//             _d_temp2 += _r_d81 * tempWT;
//             _d_tempWT += temp2 * _r_d81;
//             _d_temp1 += _r_d81 * ((-ux + uz) * (4.5F * (-ux + uz) + 3.F) - u2);
//             _d_ux += -temp1 * _r_d81 * (4.5F * (-ux + uz) + 3.F);
//             _d_uz += temp1 * _r_d81 * (4.5F * (-ux + uz) + 3.F);
//             _d_ux += -4.5F * (-ux + uz) * temp1 * _r_d81;
//             _d_uz += 4.5F * (-ux + uz) * temp1 * _r_d81;
//             _d_u2 += -temp1 * _r_d81;
//         }
//         {
//             tempSW = _t21;
//             float _r_d80 = _d_tempSW;
//             _d_tempSW = 0.F;
//             _d_temp2 += _r_d80 * tempSW;
//             _d_tempSW += temp2 * _r_d80;
//             _d_temp1 += _r_d80 * ((-ux - uy) * (4.5F * (-ux - uy) + 3.F) - u2);
//             _d_ux += -temp1 * _r_d80 * (4.5F * (-ux - uy) + 3.F);
//             _d_uy += -temp1 * _r_d80 * (4.5F * (-ux - uy) + 3.F);
//             _d_ux += -4.5F * (-ux - uy) * temp1 * _r_d80;
//             _d_uy += -4.5F * (-ux - uy) * temp1 * _r_d80;
//             _d_u2 += -temp1 * _r_d80;
//         }
//         {
//             tempNW = _t20;
//             float _r_d79 = _d_tempNW;
//             _d_tempNW = 0.F;
//             _d_temp2 += _r_d79 * tempNW;
//             _d_tempNW += temp2 * _r_d79;
//             _d_temp1 += _r_d79 * ((-ux + uy) * (4.5F * (-ux + uy) + 3.F) - u2);
//             _d_ux += -temp1 * _r_d79 * (4.5F * (-ux + uy) + 3.F);
//             _d_uy += temp1 * _r_d79 * (4.5F * (-ux + uy) + 3.F);
//             _d_ux += -4.5F * (-ux + uy) * temp1 * _r_d79;
//             _d_uy += 4.5F * (-ux + uy) * temp1 * _r_d79;
//             _d_u2 += -temp1 * _r_d79;
//         }
//         {
//             tempEB = _t19;
//             float _r_d78 = _d_tempEB;
//             _d_tempEB = 0.F;
//             _d_temp2 += _r_d78 * tempEB;
//             _d_tempEB += temp2 * _r_d78;
//             _d_temp1 += _r_d78 * ((+ux - uz) * (4.5F * (+ux - uz) + 3.F) - u2);
//             _d_ux += temp1 * _r_d78 * (4.5F * (+ux - uz) + 3.F);
//             _d_uz += -temp1 * _r_d78 * (4.5F * (+ux - uz) + 3.F);
//             _d_ux += 4.5F * (+ux - uz) * temp1 * _r_d78;
//             _d_uz += -4.5F * (+ux - uz) * temp1 * _r_d78;
//             _d_u2 += -temp1 * _r_d78;
//         }
//         {
//             tempET = _t18;
//             float _r_d77 = _d_tempET;
//             _d_tempET = 0.F;
//             _d_temp2 += _r_d77 * tempET;
//             _d_tempET += temp2 * _r_d77;
//             _d_temp1 += _r_d77 * ((+ux + uz) * (4.5F * (+ux + uz) + 3.F) - u2);
//             _d_ux += temp1 * _r_d77 * (4.5F * (+ux + uz) + 3.F);
//             _d_uz += temp1 * _r_d77 * (4.5F * (+ux + uz) + 3.F);
//             _d_ux += 4.5F * (+ux + uz) * temp1 * _r_d77;
//             _d_uz += 4.5F * (+ux + uz) * temp1 * _r_d77;
//             _d_u2 += -temp1 * _r_d77;
//         }
//         {
//             tempSE = _t17;
//             float _r_d76 = _d_tempSE;
//             _d_tempSE = 0.F;
//             _d_temp2 += _r_d76 * tempSE;
//             _d_tempSE += temp2 * _r_d76;
//             _d_temp1 += _r_d76 * ((+ux - uy) * (4.5F * (+ux - uy) + 3.F) - u2);
//             _d_ux += temp1 * _r_d76 * (4.5F * (+ux - uy) + 3.F);
//             _d_uy += -temp1 * _r_d76 * (4.5F * (+ux - uy) + 3.F);
//             _d_ux += 4.5F * (+ux - uy) * temp1 * _r_d76;
//             _d_uy += -4.5F * (+ux - uy) * temp1 * _r_d76;
//             _d_u2 += -temp1 * _r_d76;
//         }
//         {
//             tempNE = _t16;
//             float _r_d75 = _d_tempNE;
//             _d_tempNE = 0.F;
//             _d_temp2 += _r_d75 * tempNE;
//             _d_tempNE += temp2 * _r_d75;
//             _d_temp1 += _r_d75 * ((+ux + uy) * (4.5F * (+ux + uy) + 3.F) - u2);
//             _d_ux += temp1 * _r_d75 * (4.5F * (+ux + uy) + 3.F);
//             _d_uy += temp1 * _r_d75 * (4.5F * (+ux + uy) + 3.F);
//             _d_ux += 4.5F * (+ux + uy) * temp1 * _r_d75;
//             _d_uy += 4.5F * (+ux + uy) * temp1 * _r_d75;
//             _d_u2 += -temp1 * _r_d75;
//         }
//         {
//             tempSB = _t15;
//             float _r_d74 = _d_tempSB;
//             _d_tempSB = 0.F;
//             _d_temp2 += _r_d74 * tempSB;
//             _d_tempSB += temp2 * _r_d74;
//             _d_temp1 += _r_d74 * ((-uy - uz) * (4.5F * (-uy - uz) + 3.F) - u2);
//             _d_uy += -temp1 * _r_d74 * (4.5F * (-uy - uz) + 3.F);
//             _d_uz += -temp1 * _r_d74 * (4.5F * (-uy - uz) + 3.F);
//             _d_uy += -4.5F * (-uy - uz) * temp1 * _r_d74;
//             _d_uz += -4.5F * (-uy - uz) * temp1 * _r_d74;
//             _d_u2 += -temp1 * _r_d74;
//         }
//         {
//             tempST = _t14;
//             float _r_d73 = _d_tempST;
//             _d_tempST = 0.F;
//             _d_temp2 += _r_d73 * tempST;
//             _d_tempST += temp2 * _r_d73;
//             _d_temp1 += _r_d73 * ((-uy + uz) * (4.5F * (-uy + uz) + 3.F) - u2);
//             _d_uy += -temp1 * _r_d73 * (4.5F * (-uy + uz) + 3.F);
//             _d_uz += temp1 * _r_d73 * (4.5F * (-uy + uz) + 3.F);
//             _d_uy += -4.5F * (-uy + uz) * temp1 * _r_d73;
//             _d_uz += 4.5F * (-uy + uz) * temp1 * _r_d73;
//             _d_u2 += -temp1 * _r_d73;
//         }
//         {
//             tempNB = _t13;
//             float _r_d72 = _d_tempNB;
//             _d_tempNB = 0.F;
//             _d_temp2 += _r_d72 * tempNB;
//             _d_tempNB += temp2 * _r_d72;
//             _d_temp1 += _r_d72 * ((+uy - uz) * (4.5F * (+uy - uz) + 3.F) - u2);
//             _d_uy += temp1 * _r_d72 * (4.5F * (+uy - uz) + 3.F);
//             _d_uz += -temp1 * _r_d72 * (4.5F * (+uy - uz) + 3.F);
//             _d_uy += 4.5F * (+uy - uz) * temp1 * _r_d72;
//             _d_uz += -4.5F * (+uy - uz) * temp1 * _r_d72;
//             _d_u2 += -temp1 * _r_d72;
//         }
//         {
//             tempNT = _t12;
//             float _r_d71 = _d_tempNT;
//             _d_tempNT = 0.F;
//             _d_temp2 += _r_d71 * tempNT;
//             _d_tempNT += temp2 * _r_d71;
//             _d_temp1 += _r_d71 * ((+uy + uz) * (4.5F * (+uy + uz) + 3.F) - u2);
//             _d_uy += temp1 * _r_d71 * (4.5F * (+uy + uz) + 3.F);
//             _d_uz += temp1 * _r_d71 * (4.5F * (+uy + uz) + 3.F);
//             _d_uy += 4.5F * (+uy + uz) * temp1 * _r_d71;
//             _d_uz += 4.5F * (+uy + uz) * temp1 * _r_d71;
//             _d_u2 += -temp1 * _r_d71;
//         }
//         {
//             temp1 = _t11;
//             float _r_d70 = _d_temp1;
//             _d_temp1 = 0.F;
//             _d_temp_base += (1.F / 36.F) * _r_d70;
//         }
//         {
//             tempW = _t10;
//             float _r_d69 = _d_tempW;
//             _d_tempW = 0.F;
//             _d_temp2 += _r_d69 * tempW;
//             _d_tempW += temp2 * _r_d69;
//             _d_temp1 += _r_d69 * (ux * (4.5F * ux - 3.F) - u2);
//             _d_ux += temp1 * _r_d69 * (4.5F * ux - 3.F);
//             _d_ux += 4.5F * ux * temp1 * _r_d69;
//             _d_u2 += -temp1 * _r_d69;
//         }
//         {
//             tempE = _t9;
//             float _r_d68 = _d_tempE;
//             _d_tempE = 0.F;
//             _d_temp2 += _r_d68 * tempE;
//             _d_tempE += temp2 * _r_d68;
//             _d_temp1 += _r_d68 * (ux * (4.5F * ux + 3.F) - u2);
//             _d_ux += temp1 * _r_d68 * (4.5F * ux + 3.F);
//             _d_ux += 4.5F * ux * temp1 * _r_d68;
//             _d_u2 += -temp1 * _r_d68;
//         }
//         {
//             tempB = _t8;
//             float _r_d67 = _d_tempB;
//             _d_tempB = 0.F;
//             _d_temp2 += _r_d67 * tempB;
//             _d_tempB += temp2 * _r_d67;
//             _d_temp1 += _r_d67 * (uz * (4.5F * uz - 3.F) - u2);
//             _d_uz += temp1 * _r_d67 * (4.5F * uz - 3.F);
//             _d_uz += 4.5F * uz * temp1 * _r_d67;
//             _d_u2 += -temp1 * _r_d67;
//         }
//         {
//             tempT = _t7;
//             float _r_d66 = _d_tempT;
//             _d_tempT = 0.F;
//             _d_temp2 += _r_d66 * tempT;
//             _d_tempT += temp2 * _r_d66;
//             _d_temp1 += _r_d66 * (uz * (4.5F * uz + 3.F) - u2);
//             _d_uz += temp1 * _r_d66 * (4.5F * uz + 3.F);
//             _d_uz += 4.5F * uz * temp1 * _r_d66;
//             _d_u2 += -temp1 * _r_d66;
//         }
//         {
//             tempS = _t6;
//             float _r_d65 = _d_tempS;
//             _d_tempS = 0.F;
//             _d_temp2 += _r_d65 * tempS;
//             _d_tempS += temp2 * _r_d65;
//             _d_temp1 += _r_d65 * (uy * (4.5F * uy - 3.F) - u2);
//             _d_uy += temp1 * _r_d65 * (4.5F * uy - 3.F);
//             _d_uy += 4.5F * uy * temp1 * _r_d65;
//             _d_u2 += -temp1 * _r_d65;
//         }
//         {
//             tempN = _t5;
//             float _r_d64 = _d_tempN;
//             _d_tempN = 0.F;
//             _d_temp2 += _r_d64 * tempN;
//             _d_tempN += temp2 * _r_d64;
//             _d_temp1 += _r_d64 * (uy * (4.5F * uy + 3.F) - u2);
//             _d_uy += temp1 * _r_d64 * (4.5F * uy + 3.F);
//             _d_uy += 4.5F * uy * temp1 * _r_d64;
//             _d_u2 += -temp1 * _r_d64;
//         }
//         {
//             temp1 = _t4;
//             float _r_d63 = _d_temp1;
//             _d_temp1 = 0.F;
//             _d_temp_base += (1.F / 18.F) * _r_d63;
//         }
//         {
//             tempC = _t3;
//             float _r_d62 = _d_tempC;
//             _d_tempC = 0.F;
//             _d_temp2 += _r_d62 * tempC;
//             _d_tempC += temp2 * _r_d62;
//             _d_temp1 += _r_d62 * -u2;
//             _d_u2 += -temp1 * _r_d62;
//         }
//         {
//             float _r_d61 = _d_temp2;
//             _d_temp2 = 0.F;
//         }
//         {
//             float _r_d60 = _d_temp1;
//             _d_temp1 = 0.F;
//             _d_temp_base += (1.F / 3.F) * _r_d60;
//         }
//         {
//             float _r_d59 = _d_temp_base;
//             _d_temp_base = 0.F;
//             _d_rho += 1.95F * _r_d59;
//         }
//         {
//             float _r_d58 = _d_temp1;
//             _d_temp1 = 0.F;
//             _d_temp_base += (1.f / 3.f) * _r_d58;
//         }
//         {
//             float _r_d57 = _d_temp_base;
//             _d_temp_base = 0.F;
//             _d_rho += 1.95f * _r_d57;
//         }
//         {
//             float _r_d56 = _d_u2;
//             _d_u2 = 0.F;
//             _d_ux += 1.5F * _r_d56 * ux;
//             _d_ux += ux * 1.5F * _r_d56;
//             _d_uy += 1.5F * _r_d56 * uy;
//             _d_uy += uy * 1.5F * _r_d56;
//             _d_uz += 1.5F * _r_d56 * uz;
//             _d_uz += uz * 1.5F * _r_d56;
//         }
//         if (_cond1) {
//             {
//                 float _r_d55 = _d_uz;
//                 _d_uz = 0.F;
//             }
//             {
//                 float _r_d54 = _d_uy;
//                 _d_uy = 0.F;
//             }
//             {
//                 float _r_d53 = _d_ux;
//                 _d_ux = 0.F;
//             }
//         }
//         {
//             uz = _t2;
//             float _r_d52 = _d_uz;
//             _d_uz = 0.F;
//             _d_uz += _r_d52 / rho;
//             float _r2 = _r_d52 * -uz / (rho * rho);
//             _d_rho += _r2;
//         }
//         {
//             uy = _t1;
//             float _r_d51 = _d_uy;
//             _d_uy = 0.F;
//             _d_uy += _r_d51 / rho;
//             float _r1 = _r_d51 * -uy / (rho * rho);
//             _d_rho += _r1;
//         }
//         {
//             ux = _t0;
//             float _r_d50 = _d_ux;
//             _d_ux = 0.F;
//             _d_ux += _r_d50 / rho;
//             float _r0 = _r_d50 * -ux / (rho * rho);
//             _d_rho += _r0;
//         }
//         {
//             float _r_d49 = _d_uz;
//             _d_uz = 0.F;
//             _d_tempT += _r_d49;
//             _d_tempB += -_r_d49;
//             _d_tempNT += _r_d49;
//             _d_tempNB += -_r_d49;
//             _d_tempST += _r_d49;
//             _d_tempSB += -_r_d49;
//             _d_tempET += _r_d49;
//             _d_tempEB += -_r_d49;
//             _d_tempWT += _r_d49;
//             _d_tempWB += -_r_d49;
//         }
//         {
//             float _r_d48 = _d_uy;
//             _d_uy = 0.F;
//             _d_tempN += _r_d48;
//             _d_tempS += -_r_d48;
//             _d_tempNE += _r_d48;
//             _d_tempNW += _r_d48;
//             _d_tempSE += -_r_d48;
//             _d_tempSW += -_r_d48;
//             _d_tempNT += _r_d48;
//             _d_tempNB += _r_d48;
//             _d_tempST += -_r_d48;
//             _d_tempSB += -_r_d48;
//         }
//         {
//             float _r_d47 = _d_ux;
//             _d_ux = 0.F;
//             _d_tempE += _r_d47;
//             _d_tempW += -_r_d47;
//             _d_tempNE += _r_d47;
//             _d_tempNW += -_r_d47;
//             _d_tempSE += _r_d47;
//             _d_tempSW += -_r_d47;
//             _d_tempET += _r_d47;
//             _d_tempEB += _r_d47;
//             _d_tempWT += -_r_d47;
//             _d_tempWB += -_r_d47;
//         }
//         {
//             float _r_d46 = _d_rho;
//             _d_rho = 0.F;
//             _d_tempC += _r_d46;
//             _d_tempN += _r_d46;
//             _d_tempS += _r_d46;
//             _d_tempE += _r_d46;
//             _d_tempW += _r_d46;
//             _d_tempT += _r_d46;
//             _d_tempB += _r_d46;
//             _d_tempNE += _r_d46;
//             _d_tempNW += _r_d46;
//             _d_tempSE += _r_d46;
//             _d_tempSW += _r_d46;
//             _d_tempNT += _r_d46;
//             _d_tempNB += _r_d46;
//             _d_tempST += _r_d46;
//             _d_tempSB += _r_d46;
//             _d_tempET += _r_d46;
//             _d_tempEB += _r_d46;
//             _d_tempWT += _r_d46;
//             _d_tempWB += _r_d46;
//         }
//     }
//     {
//         float _r_d18 = _d_tempWB;
//         _d_tempWB = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WB + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d18);
//     }
//     {
//         float _r_d17 = _d_tempWT;
//         _d_tempWT = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * WT + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d17);
//     }
//     {
//         float _r_d16 = _d_tempEB;
//         _d_tempEB = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * EB + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d16);
//     }
//     {
//         float _r_d15 = _d_tempET;
//         _d_tempET = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ET + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d15);
//     }
//     {
//         float _r_d14 = _d_tempSB;
//         _d_tempSB = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SB + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d14);
//     }
//     {
//         float _r_d13 = _d_tempST;
//         _d_tempST = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * ST + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d13);
//     }
//     {
//         float _r_d12 = _d_tempNB;
//         _d_tempNB = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NB + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d12);
//     }
//     {
//         float _r_d11 = _d_tempNT;
//         _d_tempNT = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NT + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d11);
//     }
//     {
//         float _r_d10 = _d_tempSW;
//         _d_tempSW = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SW + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d10);
//     }
//     {
//         float _r_d9 = _d_tempSE;
//         _d_tempSE = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * SE + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d9);
//     }
//     {
//         float _r_d8 = _d_tempNW;
//         _d_tempNW = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NW + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d8);
//     }
//     {
//         float _r_d7 = _d_tempNE;
//         _d_tempNE = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * NE + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d7);
//     }
//     {
//         float _r_d6 = _d_tempB;
//         _d_tempB = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * B + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d6);
//     }
//     {
//         float _r_d5 = _d_tempT;
//         _d_tempT = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * T + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d5);
//     }
//     {
//         float _r_d4 = _d_tempW;
//         _d_tempW = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * W + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d4);
//     }
//     {
//         float _r_d3 = _d_tempE;
//         _d_tempE = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * E + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d3);
//     }
//     {
//         float _r_d2 = _d_tempS;
//         _d_tempS = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * S + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d2);
//     }
//     {
//         float _r_d1 = _d_tempN;
//         _d_tempN = 0.F;
//         atomicAdd(&_d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * N + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))], _r_d1);
//     }
//     {
//         float _r_d0 = _d_tempC;
//         _d_tempC = 0.F;
//         _d_srcGrid[(((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0)))] += _r_d0;
//         if ((((120 + 8) * (120 + 0) * (150 + 4)) * C + ((0 + __temp_x__0) + (0 + __temp_y__0) * (120 + 8) + (0 + __temp_z__0) * (120 + 8) * (120 + 0))) == 15489)
//             printf("_d_srcGrid: %f\n", _d_srcGrid[15489]);
//     }
// }
// void CUDA_LBM_performStreamCollide_grad(float *srcGrid, float *dstGrid, float *_d_srcGrid, float *_d_dstGrid) {
//     dim3 dimBlock(1, 1, 1), dimGrid(1, 1, 1);
//     dimBlock.x = (120);
//     dimGrid.x = (120);
//     dimGrid.y = (150);
//     dimBlock.y = dimBlock.z = dimGrid.z = 1;
//     performStreamCollide_kernel<<<dimGrid, dimBlock>>>(srcGrid, dstGrid);
//     performStreamCollide_kernel_pullback<<<dimGrid, dimBlock>>>(srcGrid, dstGrid, _d_srcGrid, _d_dstGrid);
// }