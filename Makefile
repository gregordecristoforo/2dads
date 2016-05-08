all: main

main: slab output 
	nvcc $(CUDACFLAGS)  main.cu slab.o output.o $(INCLUDES)  $(LFLAGS) -o 2dads -lcufft -lcusolver -lcusparse -lhdf5 -lhdf5_cpp -lhdf5_hl_cpp -lhdf5_hl -lpthread -lz -ldl
	
slab:
	g++ $(CFLAGS) $(OPTS) $(INCLUDES) -c -o slab.o slab.cpp

output:
	g++ $(CFLAGS) $(OPTS) $(INCLUDES) -c -o output.o output.cpp

clean:
	rm 2dads

CUDACFLAGS	= -O2 -std c++11 --gpu-architecture sm_30 --compiler-options -Wall
CFLAGS = -O2 -march=native -std=c++11 -Wall -malign-double
LFLAGS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial
INCLUDES =  -I/usr/include/hdf5/serial
OPTS = -std=c++11 -O0


