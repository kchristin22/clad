#===============================================================================
# User Options
#===============================================================================

CUDA_PATH ?= /usr/local/cuda-11.8

CLAD_PATH ?= ../../build/./lib/clad.so
CLANG_PATH ?= /usr/lib/llvm-19/bin/clang++

OPTIMIZE    ?= yes

NEWCACHE    ?= yes
AA          ?= no
PHISTRUCT   ?= no
FORWARD     ?= no
COALESE     ?= yes
CACHELICM   ?= yes
ABI         ?= yes
ALLOCATOR   ?= yes
VERIFY      ?= yes

DEBUG       = no
PROFILE     = no
SM_VERSION  = 75

#===============================================================================
# Program name & source code list
#===============================================================================

program = lbm

source = lbm.cu main.cc parboil_cuda.c args.c

obj = $(source:.cu=.o)
#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -mllvm -max-heap-to-stack-size=1000000 -I $(CUDA_PATH)/include -I .
# -mllvm -max-heap-to-stack-size=-1
# -Rpass=attributor -mllvm -debug -mllvm -debug-only=attributor


CC := $(CLANG_PATH)
CFLAGS += -ffast-math --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION) \
          -std=c++17 -Xclang -add-plugin -Xclang clad -Xclang -plugin-arg-clad -Xclang -fgenerate-source-file -Xclang -plugin-arg-clad -Xclang -enable-tbr -Xclang -load -Xclang $(CLAD_PATH) -I../../include
  

# Linker Flags
LDFLAGS = -lcudart_static -ldl -lrt -lpthread -lm -lstdc++ 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS += -g -G
endif

ifeq ($(ALLOCATOR),yes)
  CFLAGS += -DALLOCATOR
endif

ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY
endif

ifeq ($(ABI),yes)
  CFLAGS += -DABI
else
  CFLAGS += -DSIZE=8
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


# Optimization Flags
ifeq ($(FORWARD),yes)
  CFLAGS += 
else
  CFLAGS += -DALLOW_AD=1 
endif


# Optimization Flags
# ifeq ($(NEWCACHE),yes)
#   CFLAGS += -mllvm -enzyme-new-cache=1 -mllvm -enzyme-mincut-cache=1
#   ifeq ($(ABI),yes)
#     CFLAGS += -DSIZE=20
#   endif
# else
#   CFLAGS += -mllvm -enzyme-new-cache=0 -mllvm -enzyme-mincut-cache=0
#   ifeq ($(ABI),yes)
#     CFLAGS += -DSIZE=80
#   endif
# endif

# ifeq ($(AA),yes)
#   CFLAGS += -mllvm -enzyme-aggressive-aa=1
# else
#   CFLAGS += -mllvm -enzyme-aggressive-aa=0
# endif


# ifeq ($(PHISTRUCT),yes)
#   CFLAGS += -mllvm -enzyme-phi-restructure=1
# else
#   CFLAGS += -mllvm -enzyme-phi-restructure=0
# endif

# ifeq ($(COALESE),yes)
#   CFLAGS += -mllvm -enzyme-coalese
# endif

# ifeq ($(CACHELICM),yes)
#   CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=1
# else
#   CFLAGS +=  -mllvm -enzyme-loop-invariant-cache=0
# endif



#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) lbm_kernel.cu Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cc lbm_kernel.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu lbm_kernel.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -rf lbm main.o lbm.o Derivatives.cpp

run:
	./lbm
