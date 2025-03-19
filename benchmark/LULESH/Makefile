# Environment Variables
CUDA_PATH ?= /usr/local/cuda-11.8
# ENZYME_PATH ?= /home/lpaehler/Work/temporary_files/Enzyme/build/ClangEnzyme-13.so
CLAD_PATH ?= ../../build/./lib/clad.so
CLANG_PATH ?= /usr/lib/llvm-19/bin/clang++
SM_VERSION  = 75

# Ablation testing & numerical verification
VERIFY  ?= no
FORWARD ?= no
BRANCHYOPT ?= no
OPTIMIZE ?= yes
RESTRICT ?= yes


DEBUG ?= no

CC := $(CLANG_PATH) 
CFLAGS  = -std=c++17 --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION) \
		  -Xclang -add-plugin -Xclang clad -Xclang -plugin-arg-clad -Xclang -enable-tbr -Xclang -plugin-arg-clad -Xclang -fgenerate-source-file -Xclang -load -Xclang $(CLAD_PATH) -I../../include
LDFLAGS = -lcudart_static -ldl -lrt -lpthread -lm -lstdc++ 


ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3 
endif

ifeq ($(RESTRICT),yes)
	CFLAGS += -DRESTRICT=1
else
	CFLAGS += -DRESTRICT=0
endif

ifeq ($(DEBUG),yes)
	CFLAGS += -g
endif

ifeq ($(VERIFY),yes)
	CFLAGS += -DVERIFY
endif

ifeq ($(BRANCHYOPT),yes)
	CFLAGS += -DBRANCHYOPT
endif

ifeq ($(FORWARD),yes)
	CFLAGS += -DNormal_forward=1
else
	CFLAGS += -DNormal_forward=0
endif
#SILO_INCLUDES := /usr/local/silo-4.8/include
#SILO_LIBS := /usr/local/silo-4.8/lib

#LINKFLAGS = -lmpich -L$(MPICH_DIR)/lib 
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

all: release 

debug: LINKFLAGS += 

release: 	FLAGS += $(RFLAGS)
debug: 		FLAGS += $(DFLAGS)

release: lulesh
debug: lulesh

lulesh: allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o
	$(CC) $(CFLAGS) allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o -o lulesh $(LDFLAGS)

allocator.o: allocator.cu vector.h
	$(CC) $(CFLAGS) allocator.cu -I ./ -c -o allocator.o

lulesh.o: lulesh.cu util.h vector.h allocator.h
	$(CC) $(CFLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o lulesh.o

lulesh-comms.o: lulesh-comms.cu
	$(CC) $(CFLAGS) lulesh-comms.cu -I ./ -c -o lulesh-comms.o

lulesh-comms-gpu.o: lulesh-comms-gpu.cu
	$(CC) $(CFLAGS) lulesh-comms-gpu.cu -I ./ -c -o lulesh-comms-gpu.o

clean: 
	rm -rf allocator.o lulesh-comms.o lulesh-comms-gpu.o lulesh.o lulesh xyz.asc regNumList.txt