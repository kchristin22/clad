#pragma once

#include "vector.h"

#define LULESH_SHOW_PROGRESS 0
#define DOUBLE_PRECISION
//#define SAMI 

#if USE_MPI
#include <mpi.h>

/*
   define one of these three symbols:

   SEDOV_SYNC_POS_VEL_NONE
   SEDOV_SYNC_POS_VEL_EARLY
   SEDOV_SYNC_POS_VEL_LATE
*/

// TODO: currently we support only early sync!
#define SEDOV_SYNC_POS_VEL_EARLY 1
#endif

enum {
  VolumeError = -1,
  QStopError = -2,
  LFileError = -3
} ;

/* Could also support fixed point and interval arithmetic types */
typedef float        real4 ;
typedef double       real8 ;

typedef int    Index_t ; /* array subscript and loop index */
typedef int    Int_t ;   /* integer representation */
#ifdef DOUBLE_PRECISION
typedef real8  Real_t ;  /* floating point representation */
#else
typedef real4  Real_t ;  /* floating point representation */
#endif

class Domain
{

public: 

  void sortRegions(Vector_h<Int_t>& regReps_h, Vector_h<Index_t>& regSorted_h);
  void CreateRegionIndexSets(Int_t nr, Int_t balance);


  Index_t max_streams;
  std::vector<cudaStream_t> streams;

  /* Elem-centered */

  Vector_d<Index_t> matElemlist ; /* material indexset */
  Vector_d<Index_t> nodelist ;    /* elemToNode connectivity */

  Vector_d<Index_t> lxim ;        /* element connectivity through face */
  Vector_d<Index_t> lxip ;
  Vector_d<Index_t> letam ;
  Vector_d<Index_t> letap ;
  Vector_d<Index_t> lzetam ;
  Vector_d<Index_t> lzetap ;

  Vector_d<Int_t> elemBC ;        /* elem face symm/free-surf flag */

  Vector_d<Real_t> e ;            /* energy */
  Vector_d<Real_t> d_e;           /* change in energy */

  Vector_d<Real_t> p ;            /* pressure */

  Vector_d<Real_t> q ;            /* q */
  Vector_d<Real_t> ql ;           /* linear term for q */
  Vector_d<Real_t> qq ;           /* quadratic term for q */

  Vector_d<Real_t> v ;            /* relative volume */

  Vector_d<Real_t> volo ;         /* reference volume */
  Vector_d<Real_t> delv ;         /* m_vnew - m_v */
  Vector_d<Real_t> vdov ;         /* volume derivative over volume */

  Vector_d<Real_t> arealg ;       /* char length of an element */
  
  Vector_d<Real_t> ss ;           /* "sound speed" */

  Vector_d<Real_t> elemMass ;     /* mass */

  Vector_d<Real_t>* vnew ;         /* new relative volume -- temporary */

  Vector_d<Real_t>* delv_xi ;      /* velocity gradient -- temporary */
  Vector_d<Real_t>* delv_eta ;
  Vector_d<Real_t>* delv_zeta ;

  Vector_d<Real_t>* delx_xi ;      /* coordinate gradient -- temporary */
  Vector_d<Real_t>* delx_eta ;
  Vector_d<Real_t>* delx_zeta ;

  Vector_d<Real_t>* dxx ;          /* principal strains -- temporary */
  Vector_d<Real_t>* dyy ;
  Vector_d<Real_t>* dzz ;

  /* Node-centered */

  Vector_d<Real_t> x ;            /* coordinates */
  Vector_d<Real_t> y ;
  Vector_d<Real_t> z ;

  Vector_d<Real_t> xd ;           /* velocities */
  Vector_d<Real_t> yd ;
  Vector_d<Real_t> zd ;

  Vector_d<Real_t> xdd ;          /* accelerations */
  Vector_d<Real_t> ydd ;
  Vector_d<Real_t> zdd ;


  Vector_d<Real_t> fx ;           /* forces */
  Vector_d<Real_t> fy ;
  Vector_d<Real_t> fz ;

  Vector_d<Real_t> dfx ;         /* AD of the forces */
  Vector_d<Real_t> dfy ;
  Vector_d<Real_t> dfz ;

  Vector_d<Real_t> nodalMass ;    /* mass */
  Vector_h<Real_t> h_nodalMass ;    /* mass - host */

  /* device pointers for comms */
  Real_t *d_delv_xi ;      /* velocity gradient -- temporary */
  Real_t *d_delv_eta ;
  Real_t *d_delv_zeta ;

  Real_t *d_x ;            /* coordinates */
  Real_t *d_y ;
  Real_t *d_z ;

  Real_t *d_xd ;           /* velocities */
  Real_t *d_yd ;
  Real_t *d_zd ;

  Real_t *d_fx ;           /* forces */
  Real_t *d_fy ;
  Real_t *d_fz ;

  /* access elements for comms */
  Real_t& get_delv_xi(Index_t idx) { return d_delv_xi[idx] ; }
  Real_t& get_delv_eta(Index_t idx) { return d_delv_eta[idx] ; }
  Real_t& get_delv_zeta(Index_t idx) { return d_delv_zeta[idx] ; }

  Real_t& get_x(Index_t idx) { return d_x[idx] ; }
  Real_t& get_y(Index_t idx) { return d_y[idx] ; }
  Real_t& get_z(Index_t idx) { return d_z[idx] ; }

  Real_t& get_xd(Index_t idx) { return d_xd[idx] ; }
  Real_t& get_yd(Index_t idx) { return d_yd[idx] ; }
  Real_t& get_zd(Index_t idx) { return d_zd[idx] ; }

  Real_t& get_fx(Index_t idx) { return d_fx[idx] ; }
  Real_t& get_fy(Index_t idx) { return d_fy[idx] ; }
  Real_t& get_fz(Index_t idx) { return d_fz[idx] ; }

  // host access
  Real_t& get_nodalMass(Index_t idx) { return h_nodalMass[idx] ; }

  /* Boundary nodesets */

  Vector_d<Index_t> symmX ;       /* symmetry plane nodesets */
  Vector_d<Index_t> symmY ;        
  Vector_d<Index_t> symmZ ;
   
  Vector_d<Int_t> nodeElemCount ;
  Vector_d<Int_t> nodeElemStart;
  Vector_d<Index_t> nodeElemCornerList ;

  /* Parameters */

  Real_t dtfixed ;               /* fixed time increment */
  Real_t deltatimemultlb ;
  Real_t deltatimemultub ;
  Real_t stoptime ;              /* end time for simulation */
  Real_t dtmax ;                 /* maximum allowable time increment */
  Int_t cycle ;                  /* iteration count for simulation */

  Real_t* dthydro_h;             /* hydro time constraint */ 
  Real_t* d_dthydro_h;           /* AD change of the hydro time constraint */
  Real_t* dtcourant_h;           /* courant time constraint */
  Real_t* d_dtcourant_h;         /* AD of the courant time constraint */
  Index_t* bad_q_h;              /* flag to indicate Q error */
  Index_t* bad_vol_h;            /* flag to indicate volume error */

  /* cuda Events to indicate completion of certain kernels */
  cudaEvent_t time_constraint_computed;

  Real_t time_h ;               /* current time */
  Real_t deltatime_h ;          /* variable time increment */

  Real_t u_cut ;                /* velocity tolerance */
  Real_t hgcoef ;               /* hourglass control */
  Real_t qstop ;                /* excessive q indicator */
  Real_t monoq_max_slope ;
  Real_t monoq_limiter_mult ;   
  Real_t e_cut ;                /* energy tolerance */
  Real_t p_cut ;                /* pressure tolerance */
  Real_t ss4o3 ;
  Real_t q_cut ;                /* q tolerance */
  Real_t v_cut ;                /* relative volume tolerance */
  Real_t qlc_monoq ;            /* linear term coef for q */
  Real_t qqc_monoq ;            /* quadratic term coef for q */
  Real_t qqc ;
  Real_t eosvmax ;
  Real_t eosvmin ;
  Real_t pmin ;                 /* pressure floor */
  Real_t emin ;                 /* energy floor */
  Real_t dvovmax ;              /* maximum allowable volume change */
  Real_t refdens ;              /* reference density */

   Index_t m_colLoc ;
   Index_t m_rowLoc ;
   Index_t m_planeLoc ;
   Index_t m_tp ;

   Index_t&  colLoc()             { return m_colLoc ; }
   Index_t&  rowLoc()             { return m_rowLoc ; }
   Index_t&  planeLoc()           { return m_planeLoc ; }
   Index_t&  tp()                 { return m_tp ; }

  Index_t sizeX ;
  Index_t sizeY ;
  Index_t sizeZ ;
  Index_t maxPlaneSize ;
  Index_t maxEdgeSize ;

  Index_t numElem ;
  Index_t padded_numElem ; 

  Index_t numNode;
  Index_t padded_numNode ; 

  Index_t numSymmX ; 
  Index_t numSymmY ; 
  Index_t numSymmZ ; 

  Index_t octantCorner;

   // Region information
   Int_t numReg ; //number of regions (def:11)
   Int_t balance; //Load balance between regions of a domain (def: 1)
   Int_t  cost;  //imbalance cost (def: 1)
   Int_t*   regElemSize ;   // Size of region sets
   Vector_d<Int_t> regCSR;  // records the begining and end of each region
   Vector_d<Int_t> regReps; // records the rep number per region
   Vector_d<Index_t> regNumList;    // Region number per domain element
   Vector_d<Index_t> regElemlist;  // region indexset 
   Vector_d<Index_t> regSorted; // keeps index of sorted regions
   
   //
   // MPI-Related additional data
   //

   Index_t m_numRanks;
   Index_t& numRanks() { return m_numRanks ; }

   void SetupCommBuffers(Int_t edgeNodes);
   void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems, Int_t domNodes, Int_t padded_domElems, Vector_h<Real_t> &x_h, Vector_h<Real_t> &y_h, Vector_h<Real_t> &z_h, Vector_h<Int_t> &nodelist_h);

   // Used in setup
   Index_t m_rowMin, m_rowMax;
   Index_t m_colMin, m_colMax;
   Index_t m_planeMin, m_planeMax ;

#if USE_MPI   
   // Communication Work space 
   Real_t *commDataSend ;
   Real_t *commDataRecv ;

   Real_t *d_commDataSend ;
   Real_t *d_commDataRecv ;

   // Maximum number of block neighbors 
   MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners 
   MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners 
#endif

};

typedef Real_t& (Domain::* Domain_member )(Index_t) ;

// Assume 128 byte coherence
// Assume Real_t is an "integral power of 2" bytes wide
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))

// MPI Message Tags
#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

// cpu-comms
void CommRecv(Domain& domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);
void CommSend(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);
void CommSBN(Domain& domain, Int_t xferFields, Domain_member *fieldData);
void CommSyncPosVel(Domain& domain);
void CommMonoQ(Domain& domain);

// gpu-comms
void CommSendGpu(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly, cudaStream_t stream);
void CommSBNGpu(Domain& domain, Int_t xferFields, Domain_member *fieldData, cudaStream_t *streams);
void CommSyncPosVelGpu(Domain& domain, cudaStream_t *streams);
void CommMonoQGpu(Domain& domain, cudaStream_t stream);

__attribute__((device)) void
Inner_ApplyMaterialPropertiesAndUpdateVolume_kernel_grad_14(
    Index_t length, Real_t rho0, Real_t e_cut, Real_t emin,
    const Real_t *__restrict ql, const Real_t *__restrict qq,
    const Real_t *__restrict vnew, const Real_t *__restrict v, Real_t pmin,
    Real_t p_cut, Real_t q_cut, Real_t eosvmin, Real_t eosvmax,
    const Index_t *__restrict regElemlist, Real_t *__restrict e,
    const Real_t *__restrict delv, const Real_t *__restrict p,
    const Real_t *__restrict q, Real_t ss4o3, const Real_t *__restrict ss,
    Real_t v_cut, const Index_t *__restrict bad_vol, const Int_t cost,
    const Index_t *__restrict regCSR, const Index_t *__restrict regReps,
    const Index_t numReg, Real_t *_d_e);


__device__ inline real4  SQRT(real4  arg) { return sqrtf(arg) ; }
__device__ inline real8  SQRT(real8  arg) { return sqrt(arg) ; }

__device__ inline real4  CBRT(real4  arg) { return cbrtf(arg) ; }
__device__ inline real8  CBRT(real8  arg) { return cbrt(arg) ; }

__device__ __host__ inline real4  FABS(real4  arg) { return fabsf(arg) ; }
__device__ __host__ inline real8  FABS(real8  arg) { return fabs(arg) ; }

__device__ inline real4  FMAX(real4  arg1,real4  arg2) { return fmaxf(arg1,arg2) ; }
__device__ inline real8  FMAX(real8  arg1,real8  arg2) { return fmax(arg1,arg2) ; }

#define MAX(a, b) ( ((a) > (b)) ? (a) : (b))


/* Stuff needed for boundary conditions */
/* 2 BCs on each of 6 hexahedral faces (12 bits) */
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002
#define XI_M_COMM   0x00004

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010
#define XI_P_COMM   0x00020

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080
#define ETA_M_COMM  0x00100

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400
#define ETA_P_COMM  0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

#define VOLUDER(a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,dvdc)		\
{									\
  const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;			\
									\
   dvdc= 								\
     ((a1) + (a2)) * ((b0) + (b1)) - ((a0) + (a1)) * ((b1) + (b2)) +	\
     ((a0) + (a4)) * ((b3) + (b4)) - ((a3) + (a4)) * ((b0) + (b4)) -	\
     ((a2) + (a5)) * ((b3) + (b5)) + ((a3) + (a5)) * ((b2) + (b5));	\
   dvdc *= twelfth;							\
}

 __device__ void
CalcPressureForElems_device(Real_t &p_new, Real_t &bvc, Real_t &pbvc,
                            Real_t e_old, Real_t compression, Real_t vnewc,
                            Real_t pmin, Real_t p_cut, Real_t eosvmax);


__device__ 
void ApplyMaterialPropertiesForElems_device(
    Real_t eosvmin,
    Real_t eosvmax,
    #ifdef RESTRICT
    const Real_t* __restrict__ vnew,
    const Real_t *__restrict__ v,
    #else
    const Real_t*  vnew,
    const Real_t *v,
    #endif
    Real_t& vnewc,
    #ifdef RESTRICT
    const Index_t* __restrict__ bad_vol,
    #else
    const Index_t* bad_vol,
    #endif
    Index_t zn);

__device__ Index_t giveMyRegion(const Index_t *regCSR, const Index_t i,
                                       const Index_t numReg);

__device__  void CalcEnergyForElems_device(
    Real_t &p_new, Real_t &e_new, Real_t &q_new, Real_t &bvc, Real_t &pbvc,
    Real_t p_old, Real_t e_old, Real_t q_old, Real_t compression,
    Real_t compHalfStep, Real_t vnewc, Real_t work, Real_t delvc, Real_t pmin,
    Real_t p_cut, Real_t e_cut, Real_t q_cut, Real_t emin, Real_t qq, Real_t ql,
    Real_t rho0, Real_t eosvmax, Index_t length);


__device__
void CalcSoundSpeedForElems_device(Real_t vnewc,
                                   Real_t rho0,
                                   Real_t enewc,
                                   Real_t pnewc,
                                   Real_t pbvc,
                                   Real_t bvc,
                                   Real_t ss4o3,
                                   Index_t nz,
                                   #ifdef RESTRICT
                                   const Real_t *__restrict__ ss,
                                   #else
                                   const Real_t *ss,
                                   #endif
                                   Index_t iz);

__device__
void UpdateVolumesForElems_device(Index_t numElem,
                                  Real_t v_cut,
                                  const Real_t *vnew,
                                  const Real_t *v,
                                  int i);