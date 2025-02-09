#===============================================================================
# User Options
#===============================================================================
CUDA_PATH ?= /usr/local/cuda-11.8

CLAD_PATH ?= ../../build/./lib/clad.so
CLANG_PATH ?= /usr/lib/llvm-17/bin/clang++

# ENZYME_PATH ?= /home/wmoses/git/Enzyme/enzyme/build13D/Enzyme/ClangEnzyme-13.so
# CLANG_PATH ?= /mnt/sabrent/wmoses/llvm13/buildall/bin/clang++

OPTIMIZE    ?= yes

FORWARD     ?= no
INLINE      ?= yes

DEBUG       = no
PROFILE     = no
SM_VERSION  = 60

#===============================================================================
# Program name & source code list
#===============================================================================

program = rsbench

source = \
main.cu \
simulation.cu\
io.cu \
init.cu \
material.cu \
utils.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -mllvm -max-heap-to-stack-size=1000000
# -mllvm -max-heap-to-stack-size=-1
# -Rpass=attributor -mllvm -debug -mllvm -debug-only=attributor

CC := $(CLANG_PATH)
CFLAGS := --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION) \
          -std=c++17 -Xclang -add-plugin -Xclang clad -Xclang -plugin-arg-clad -Xclang -fgenerate-source-file -Xclang -load -Xclang $(CLAD_PATH) -I../../include

# Linker Flags
LDFLAGS = "-lcudart_static" "-ldl" "-lrt" -lpthread -lm -lstdc++ 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS += -g -G
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
  CFLAGS += -DFORWARD=1
endif

ifeq ($(INLINE),yes)
  CFLAGS += -DALWAYS_INLINE=1
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) rsbench.cuh Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu rsbench.cuh Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf rsbench $(obj)

edit:
	vim -p $(source) rsbench.cuh

run:
	./rsbench -m event
