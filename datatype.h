#include <stdio.h>
#include <cstddef>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include<cuda.h>
#include "error.h"
//#include "kernels.h"
#include "cucmplx.h"
#include <cufft.h>
#include<iostream>
#include<fstream>
#include <cusolverSp.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <fstream>
using namespace std;

__global__
void clean_all_cuda( cufftDoubleReal *array_device,cufftDoubleReal *array_device_deriv, uint size, uint time_step, uint Ny, uint num_gp_y, uint Mx, uint num_gp_x);

__global__
void normalize( uint time_step, cufftDoubleReal *array_device, uint Mx, uint Ny, int num_gp_x, int num_gp_y);

__global__
void deriv_real_y_cuda(cufftDoubleReal *array_device, cufftDoubleReal *array_device_deriv, uint time_step, uint Ny, double Ly, int num_gp_x, int num_gp_y , size_t size, uint Mx);

__global__
void deriv_real_x_cuda(cufftDoubleReal *array_device, cufftDoubleReal *array_device_deriv, uint time_step, uint Ny, double Ly, int num_gp_x, int num_gp_y , size_t size, uint Mx, double Lx);


__global__
void second_deriv_real_y_cuda(cufftDoubleReal *array_device, cufftDoubleReal *array_device_deriv, uint time_step, uint Ny, double Ly, int num_gp_x, int num_gp_y , size_t size, uint Mx);

__global__
void second_deriv_real_x_cuda(cufftDoubleReal *array_device, cufftDoubleReal *array_device_deriv, uint time_step, uint Ny, double Ly, int num_gp_x, int num_gp_y , size_t size, uint Mx, double Lx);

__global__
void deriv_freq_y_cuda(CuCmplx<double> *array_device_freq, CuCmplx<double> *array_device_freq_derive ,uint time_step, uint Ny, uint Mx, double Ly);


__global__
void second_deriv_freq_y_cuda(CuCmplx<double> *array_device_freq, CuCmplx<double> *array_device_freq_derive ,uint time_step, uint Ny, uint Mx, double Ly);

__global__
void dirichlet_cuda(uint time_step, double *array_device, uint Mx, uint Ny, int num_gp_x, int num_gp_y, size_t size, double value1, double value2);

__global__
void neumann_cuda(uint time_step, double *array_device_deriv, uint Mx, uint Ny, int num_gp_x, int num_gp_y, size_t size, double value1, double value2, double Lx);

__global__
void twod_QR_array_freq(CuCmplx<double> *array_device_freq ,cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, double delta_x, double bound_type_l, double bound_type_r, double BC_1, double BC_2, cuDoubleComplex r1, double delta_t, double diff_coef);

__global__
void re_arrange_array_freq(cuDoubleComplex *d_x, cuDoubleComplex *d_x1, uint time_step, uint Ny, uint Mx);


__global__
void twod_QR_array_real(double* array_device, double* d_b, uint time_step, uint Ny, uint Mx , uint num_gp_x, uint num_gp_y, double dx);

__global__
void rhs(CuCmplx<double>* array_device, cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, CuCmplx<double>* array_arakawa_freq, CuCmplx<double>* array_device_freq_deriv, double delta_t, int order, double delta_x,double bound_type_l, double bound_type_r, double BC_1, double BC_2 , cuDoubleComplex k);

__global__
void rhs_1(CuCmplx<double>* array_device, cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, CuCmplx<double>* array_arakawa_freq, CuCmplx<double>* array_device_freq_deriv, double delta_t, int order, double delta_x,double bound_type_l, double bound_type_r, double BC_1, double BC_2 , cuDoubleComplex k);

__global__
void rhs_2(CuCmplx<double>* array_device, cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, CuCmplx<double>* array_arakawa_freq, CuCmplx<double>* array_device_freq_deriv, double delta_t, int order, double delta_x,double bound_type_l, double bound_type_r, double BC_1, double BC_2 , cuDoubleComplex k);


__global__
void time_int_reorder(double* array_device, double* d_b, uint time_step, uint Ny, uint Mx, uint num_gp_x, uint num_gp_y);

__global__
void re_arrange_array_real(double* d_x, double* d_x1, uint time_step, uint Ny, uint Mx , uint num_gp_x, uint num_gp_y);
	
__global__
void BC_array_device(double* array_device, uint time_step, uint Mx, uint Ny, uint num_gp_x, uint num_gp_y, double BC1, double BC2, double Lx);	

__global__
void prepare_gp_kernel(double* array_device, uint time_step, uint Mx, uint Ny, uint num_gp_x, uint num_gp_y, double bound_type_l, double bound_type_r, double BC1, double BC2, double Lx, double delta_x);

__global__
void arakawa_kernel(double* a, double* b,double* array_arakawa , double dx, double dy, uint time_step, uint Mx, uint Ny, uint num_gp_x, uint num_gp_y);

__global__
void rhs_without_deriv(CuCmplx<double>* array_device_freq ,cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, CuCmplx<double>* array_arakawa_freq, CuCmplx<double>* array_device_freq_deriv, double delta_t, int order, double delta_x, double bound_type_l,double bound_type_r, double BC_1, double BC_2, cuDoubleComplex r);
	
__global__
void gp_device(double* array_device, uint time_step, uint Mx, uint Ny, uint num_gp_x, uint num_gp_y,double bound_type_l, double bound_type_r, double BC1, double BC2, double Lx, double delta_x, cuDoubleComplex r);
	


class cuda_array{
	private://size of data array
		size_t size;
	
		double Lx;
		double Ly;
		
	//public:	
		cufftHandle plan_f;
		cufftHandle plan_b;
		
		cuDoubleComplex *d_b;
		cusolverSpHandle_t cusolverH;
		csrqrInfo_t info;
		csrqrInfo_t info_QR;
		cusparseMatDescr_t descrA;
		int *d_csrRowPtrA;
		int *d_csrColIndA;
		
		cuDoubleComplex *d_csrValA;
		cuDoubleComplex *d_x;
		cuDoubleComplex *d_x1; 
				
		size_t size_qr;
		size_t size_internal;
		void *buffer_qr; // working space for numerical factorization
		void *buffer_qr_QR;
		int m = Mx ;
		cusparseStatus_t cusparse_status;
		cusolverStatus_t cusolver_status;
		cusolverStatus_t cusolver_status_QR;
		int *csrRowPtrA;
		int *csrColIndA;
		int *csrRowPtrA_QR;
		int *csrColIndA_QR;

		cuDoubleComplex *csrValA;	
		cuDoubleComplex *csrValABatch;
		cuDoubleComplex *bBatch;
		cuDoubleComplex *xBatch;
		cuDoubleComplex *csrValA_QR;	
		cuDoubleComplex *csrValABatch_QR;
		//int counter;
		cuDoubleComplex r;
		cuDoubleComplex *QR_values;	
	public:
		double BC_1;
		double BC_2;
		double BC1;
		double BC2;
		double bound_type_l;
		double bound_type_r;
	
		int num_gp_x;
		int num_gp_y;
		uint time_step;
		uint Mx; // without ghost points
		uint Ny; //without ghost points
		double delta_t;
		double diff_coef;

		double *array_host;
		double* array_arakawa;
		CuCmplx<double>* array_arakawa_freq;
		cufftDoubleReal *array_device;
		CuCmplx<double> *array_device_freq;
		cufftDoubleReal *array_device_deriv;
		CuCmplx<double> *array_device_freq_deriv;
		/////////////////////////////////////////////////////
		
		int nnzA;
		int nnzA_QR;
		int batchSize;
		
		//////////////////////////////////////////////////////
		double delta_x;
		double delta_y;
		cuda_array(uint, uint, double, double, int, int, double, double, double, double, double, double); //constructor
		void FFT_forward( uint); //FFT in y-direction at time_step
		void FFT_backward( uint);
		void FFT_backward_deriv( uint);
		void FFT_arakawa( uint);
		void alloc_host_real(); //alloc at host_for_position_space
		void alloc_device_real(); //aloc at device_for_position_space
		void alloc_device_freq(); //alloc at device_for_frequency_space
		void alloc_device_deriv();
		void alloc_device_freq_deriv();
		void alloc_arakawa();
		void alloc_arakawa_freq();
		void arakawa(uint, double* , double* );
		void copy_host_to_device( uint);
		void copy_device_to_host( uint);
		void copy_arakawa_to_host( uint);
		void copy_device_deriv_to_host( uint);
//		__global__ void normalize(uint);
		void clean_all(uint);
		void deriv_real_y( uint , cufftDoubleReal*);
		void second_deriv_real_y( uint);
		void deriv_real_x( uint );
		void second_deriv_real_x( uint);
		void deriv_freq_y( uint, CuCmplx<double>*);
		void second_deriv_freq_y( uint);
		void QR_factorisation_1d( uint);
		void QR_factorisation_2d( uint, CuCmplx<double>*);
		void diffusion_x( uint);
		void diffusion_2d( uint);
		void ssti( uint);
		void ssti_without_deriv( uint);
		void ssti_start( uint, uint, uint);
		void dirichlet( uint, double, double);
		void neumann( uint, double, double);
		void fill_array_test(uint, double);
		void prepare_gp( uint);
		double* return_pointer_host(){return array_host;}
		//size_t get_size(){return 4*(Mx+num_gp_x)*(Ny+num_gp_y);}
		size_t get_size(){return time_step*(Mx+num_gp_x)*(Ny+num_gp_y);}
		int adress(uint, uint,uint);
		~cuda_array(){delete array_host;cudaFree(array_device);cudaFree(array_arakawa);cudaFree(array_arakawa_freq);cudaFree(array_device_freq);cudaFree(array_device_deriv);cudaFree(array_device_freq_deriv);cufftDestroy(plan_f);cufftDestroy(plan_b); //dectructor

free(csrRowPtrA);
free(csrColIndA);
free(csrValA);
free(csrValABatch);
free(bBatch);
free(xBatch);

cudaFree(d_csrValA);
cudaFree(d_csrRowPtrA);
cudaFree(d_csrColIndA);
cudaFree(d_x);
cudaFree(d_x1);
cudaFree(d_b);
cudaFree(buffer_qr);


cudaFree(QR_values);
/*
cudaFree(d_csrValA_QR);
cudaFree(d_csrRowPtrA_QR);
cudaFree(d_csrColIndA_QR);
//cudaFree(d_x_QR);
//cudaFree(d_x1_QR);
//cudaFree(d_b_QR);
cudaFree(buffer_qr_QR);
*/
}
};

cuda_array::cuda_array( uint b, uint c, double d, double e, int f, int g, double h, double i, double bound_l,double bound_r, double BC1, double BC2){
	//time_step = a;
	Mx = b;
	Ny = c;
	Lx = d;
	Ly = e;
	num_gp_x = f;
	num_gp_y = g;
	delta_t = h;
	diff_coef = i;
	bound_type_l = bound_l;
	bound_type_r = bound_r;
	BC_1 = BC1;
	BC_2 = BC2;
	//printf("BC_1 in constructor is %f\n", BC_1);
	delta_x = Lx/((double)Mx);
	delta_y = Ly/((double)Ny);
	//printf("BC_1 in constructor is %f\n",delta_x) ;
	
	size = get_size();
	alloc_host_real();
	alloc_device_real();
	alloc_device_freq();
	alloc_device_deriv();
	alloc_device_freq_deriv();
	alloc_arakawa();
	alloc_arakawa_freq();

	/////////////////////////////////////
	
	nnzA = Mx*3-2;
	batchSize = Ny/2 +1;
	/////////////////////////////////////

	//FFT forward
	int n_f[1] = {(int)Ny};
	int istride_f = 1;
	int idist_f = Ny+num_gp_y;
	int inembed_f[1] = {(int)Ny}; // pointer that indicates storage dimensions of input data
	int onembed_f[1] = {((int)Ny / 2 + 1)}; // pointer that indicates storage dimensions of output data
	int ostride_f = 1;
	int odist_f = ((int)Ny ) / 2 + 1;

	if(cufftPlanMany(&plan_f,1,n_f, inembed_f, istride_f, idist_f, onembed_f, ostride_f, odist_f, CUFFT_D2Z, Mx
	
	) != CUFFT_SUCCESS) {
	fprintf(stderr, "CUFFT error: Plan creation failed\n");	
	}
	
	//FFT backward

	int n_b[1] = {(int)Ny};
	int istride_b = 1;
	int odist_b = Ny + num_gp_y;
	int onembed_b[1] = {(int)Ny}; // pointer that indicates storage dimensions of input data
	int inembed_b[1] = {(int)Ny / 2 + 1}; // pointer that indicates storage dimensions of output data
	int ostride_b = 1;
	int idist_b = (int)Ny / 2 + 1;

	if(cufftPlanMany(&plan_b,1,n_b, inembed_b, istride_b, idist_b, onembed_b, ostride_b, odist_b, CUFFT_Z2D, Mx	
	
	) != CUFFT_SUCCESS) {
	fprintf(stderr, "CUFFT error: Plan creation failed");	
	}

///////////////////////////////////////////////////////////////////////////////////////////////////////
	//ssti
	csrValABatch = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA*batchSize);
	bBatch       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m*batchSize);
	xBatch       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m*batchSize);
	assert( NULL != csrValABatch );
	assert( bBatch != NULL );
	assert( NULL != xBatch );
	
	cusolverH = NULL;
	info = NULL;
	descrA = NULL;
	
	cusparse_status = CUSPARSE_STATUS_SUCCESS;
	cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;
	
	
	d_csrRowPtrA = NULL;
	d_csrColIndA = NULL;
	d_csrValA = NULL;
	//cuDoubleComplex *d_b = NULL; // batchSize * m
	d_x = NULL; // batchSize * m
	d_x1 = NULL; 
	
	
	size_qr = 0;
	size_internal = 0;
	buffer_qr = NULL; // working space for numerical factorization
	m = Mx ;
	//const int nnzA = Mx*3-2;
	
	csrRowPtrA = (int*)malloc(sizeof(int)*(m+1));
	csrRowPtrA[0]=1;
	for( int i = 1; i< m; i++){
		csrRowPtrA[i] = 3*i;
	}
	csrRowPtrA[m] = nnzA +1 ;
	csrColIndA = (int*)malloc(sizeof(int)*(nnzA));
	csrColIndA[0]=1;
	csrColIndA[1]=2;

	for( int i = 1; i< m-1;i++ ){
		csrColIndA[3*i-1] =i-1+1;
		csrColIndA[3*i] = i-1+2;
		csrColIndA[3*i+1] = i-1+3;
	}
	csrColIndA[nnzA-2] = m-1;
	csrColIndA[nnzA-1] = m;
	
	csrValA = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA);
	
	if(bound_type_l == 0){csrValA[0]= make_cuDoubleComplex(-3 ,0);}
	else{csrValA[0]= make_cuDoubleComplex(-1 ,0);}
	
	csrValA[1]= make_cuDoubleComplex(1 ,-1);

	for( int i = 1; i< m-1;i++ ){
		csrValA[3*i-1] = make_cuDoubleComplex(1.0 ,1);
		csrValA[3*i] = make_cuDoubleComplex(-2.0 ,0);
		csrValA[3*i+1] = make_cuDoubleComplex(1.0,-1);
	}
	csrValA[nnzA-2] = make_cuDoubleComplex(1.0 ,1);
	
	if(bound_type_r == 0){csrValA[nnzA-1] = make_cuDoubleComplex(-3.0 ,0);}
	else{csrValA[nnzA-1] = make_cuDoubleComplex(-1.0 ,0);}

	cuDoubleComplex cons = make_cuDoubleComplex((2*M_PI/Ly)*(2*M_PI/Ly),0);
	r = make_cuDoubleComplex(diff_coef*delta_t/(delta_x*delta_x), 0);
     
	//double inv_dx = 0.5/delta_x;
	double a0;
	//if(time_step == 0){a0 = 1.0;}	
	//if(time_step == 1){a0 = 3.0/2.0;}
	a0 = 11.0/6.0;
	
	//double inv_diff_coef = 1/diff_coef;
	for(int colidx = 0 ; colidx < nnzA ; colidx++){
		cuDoubleComplex Areg = csrValA[colidx];
        	for (int batchId = 0 ; batchId < batchSize ; batchId++){
            //double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
		if(Areg.x != 1){
	    //printf("i: %d k: %f\n", batchId, (double)(batchId%(Ny/2+1)));
		cuDoubleComplex k = make_cuDoubleComplex((batchId % (Ny/2+1)),0);
	       	csrValABatch[batchId*nnzA + colidx].x = a0+ cons.x *k.x*k.x*delta_t*diff_coef  - Areg.x*r.x;//      -  Areg.x *inv_dx_2;//advection part
		csrValABatch[batchId*nnzA + colidx].y = 0;//Areg.y;
		//counter ++;	
	    }
	    else{
	    csrValABatch[batchId*nnzA + colidx].x = Areg.x * (-1.0)*r.x;//    +0.03*Areg.y*inv_dx ; //advection part
	    csrValABatch[batchId*nnzA + colidx].y = 0;//Areg.y;
		}  
		}
    	}

	// step 2: create cusolver handle, qr info and matrix descriptor
	cusolver_status = cusolverSpCreate(&cusolverH);
	assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cusparse_status = cusparseCreateMatDescr(&descrA); 
	assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); // base-1

	cusolver_status = cusolverSpCreateCsrqrInfo(&info);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	// step 3: copy Aj and bj to device
	cudaStat1 = cudaMalloc ((void**)&d_csrValA   , sizeof(cuDoubleComplex) * nnzA * batchSize);
	cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA);
	cudaStat3 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (m+1));
	cudaStat4 = cudaMalloc ((void**)&d_b         , sizeof(cuDoubleComplex) * m * batchSize);
	cudaStat5 = cudaMalloc ((void**)&d_x         , sizeof(cuDoubleComplex) * m * batchSize);
	cudaStat6 = cudaMalloc ((void**)&d_x1         , sizeof(cuDoubleComplex) * m * batchSize);
   
	assert(cudaStat1 == cudaSuccess);
	assert(cudaStat2 == cudaSuccess);
	assert(cudaStat3 == cudaSuccess);
	assert(cudaStat4 == cudaSuccess);
	assert(cudaStat5 == cudaSuccess);
	assert(cudaStat6 == cudaSuccess);

	cudaStat1 = cudaMemcpy(d_csrValA   , csrValABatch, sizeof(cuDoubleComplex) * nnzA * batchSize, cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m+1), cudaMemcpyHostToDevice);
	//    cudaStat4 = cudaMemcpy(d_b, bBatch, sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);
	assert(cudaStat1 == cudaSuccess);
	assert(cudaStat2 == cudaSuccess);
	assert(cudaStat3 == cudaSuccess);
	assert(cudaStat4 == cudaSuccess);



	// step 4: symbolic analysis
	cusolver_status = cusolverSpXcsrqrAnalysisBatched(
		cusolverH, m, m, nnzA,
		descrA, d_csrRowPtrA, d_csrColIndA,
		info);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	// step 5: prepare working space
	cusolver_status = cusolverSpZcsrqrBufferInfoBatched(
		cusolverH, m, m, nnzA,
		descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
		batchSize,
		info,
		&size_internal,
		&size_qr);
		
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);      
	printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr);      


	cudaStat1 = cudaMalloc((void**)&buffer_qr, size_qr);
	assert(cudaStat1 == cudaSuccess);


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////QR_Fact
	/*
	//csrValABatch = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA*batchSize);
	bBatch       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m*batchSize);
	xBatch       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m*batchSize);
	assert( NULL != csrValABatch );
	assert( bBatch != NULL );
	assert( NULL != xBatch );
	
	cusolverH = NULL;
	info = NULL;
	descrA = NULL;
	
	cusparse_status = CUSPARSE_STATUS_SUCCESS;
	cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;
	
	
	d_csrRowPtrA = NULL;
	d_csrColIndA = NULL;
	d_csrValA = NULL;
	//cuDoubleComplex *d_b = NULL; // batchSize * m
	d_x = NULL; // batchSize * m
	d_x1 = NULL; 
	
	
	size_qr = 0;
	size_internal = 0;
	buffer_qr = NULL; // working space for numerical factorization
	m = Mx ;
	*/
	
	cudaError_t cudaStat1_QR = cudaSuccess;
	
	int m= Mx;
	    const int nnzA_QR = Mx*3-2;
	//    const int csrRowPtrA[m+1]  = { 1, 3, 6,Mx - 1 };
	
		int *csrRowPtrA_QR;

		csrRowPtrA_QR = (int*)malloc(sizeof(int)*(m+1));
		//printf("csrRowPtrA_QR adress = %p\n", csrRowPtrA_QR);
	
		csrRowPtrA_QR[0]=1;
		for( int i = 1; i< m; i++){
			csrRowPtrA_QR[i] = 3*i;
		}
		csrRowPtrA_QR[m] = nnzA +1 ;
	//const int csrColIndA[nnzA] = { 1, 3, 2, 3};
 		int *csrColIndA_QR;
		csrColIndA_QR = (int*)malloc(sizeof(int)*(nnzA));
		csrColIndA_QR[0]=1;
		csrColIndA_QR[1]=2;

		for( int i = 1; i< m-1;i++ ){
			csrColIndA_QR[3*i-1] =i-1+1;
			csrColIndA_QR[3*i] = i-1+2;
			csrColIndA_QR[3*i+1] = i-1+3;
		}
		csrColIndA_QR[nnzA-2] = m-1;
		csrColIndA_QR[nnzA-1] = m;
	    	cuDoubleComplex *csrValA_QR;
		csrValA_QR = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA_QR);
	
	//double dx_2 = (Lx/Mx)*(Lx/Mx);
		double inv_dx_2_QR = (Mx/Lx)*(Mx/Lx);

		if(bound_type_l == 0){csrValA_QR[0]= make_cuDoubleComplex(-3 ,0);}
		else{csrValA_QR[0]= make_cuDoubleComplex(-1 ,0);}
			
		csrValA_QR[1]= make_cuDoubleComplex(1 ,0);

		for( int i = 1; i< m-1;i++ ){
			csrValA_QR[3*i-1] = make_cuDoubleComplex(1.0 ,0);
			csrValA_QR[3*i] = make_cuDoubleComplex(-2.0 ,0);
			csrValA_QR[3*i+1] = make_cuDoubleComplex(1.0,0);
		}
		csrValA_QR[nnzA-2] = make_cuDoubleComplex(1.0 ,0);
		
		if(bound_type_r == 0){csrValA_QR[nnzA-1] = make_cuDoubleComplex(-3.0 ,0);}
		else{csrValA_QR[nnzA-1] = make_cuDoubleComplex(-1.0 ,0);}
   
	    const int batchSize = Ny/2 +1;

	    cuDoubleComplex *csrValABatch_QR = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA*batchSize);
	   //ssti
	   cuDoubleComplex cons_QR = make_cuDoubleComplex((2*M_PI/Ly)*(2*M_PI/Ly),0);
	
		//int counter = 0;
	    for(int colidx = 0 ; colidx < nnzA ; colidx++){
	        cuDoubleComplex Areg_QR = csrValA_QR[colidx];
	        for (int batchId = 0 ; batchId < batchSize ; batchId++){
	            //double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
		    if(Areg_QR.x != 1){
		    //printf("i: %d k: %f\n", batchId, (double)(batchId%(Ny/2+1)));
		    cuDoubleComplex k_QR = make_cuDoubleComplex((batchId % (Ny/2+1)),0);
	       csrValABatch_QR[batchId*nnzA_QR + colidx].x = Areg_QR.x*inv_dx_2_QR - cons_QR.x *k_QR.x*k_QR.x;
		    csrValABatch_QR[batchId*nnzA_QR + colidx].y = Areg_QR.y;
			//counter ++;	
		    }
		    else{
		    csrValABatch_QR[batchId*nnzA_QR + colidx].x = Areg_QR.x*inv_dx_2_QR;
		    csrValABatch_QR[batchId*nnzA_QR + colidx].y = Areg_QR.y;
		}  
		}
	    }

	cudaMalloc((void**)&QR_values,  sizeof(cuDoubleComplex) * (Mx*3-2) * batchSize);

    
	cudaStat1_QR = cudaMemcpy(QR_values   , csrValABatch_QR, sizeof(cuDoubleComplex) * nnzA_QR * batchSize, cudaMemcpyHostToDevice);
	assert(cudaStat1_QR == cudaSuccess);
	
	free(csrRowPtrA_QR);
	free(csrColIndA_QR);
	free(csrValA_QR);
	free(csrValABatch_QR);
}

void cuda_array::arakawa( uint time_step, double* first_array ,double* second_array){
	arakawa_kernel<<< ((Ny+num_gp_y)*(Mx+num_gp_x)+63)/64,64>>>( first_array, second_array, array_arakawa, delta_x, delta_y, time_step, Mx, Ny, num_gp_x, num_gp_y);
}

void cuda_array::prepare_gp( uint time_step){
	prepare_gp_kernel<<< ((Ny+num_gp_y)*(Mx+num_gp_x)+63)/64, 64>>>( array_device, time_step, Mx, Ny,num_gp_x, num_gp_y,bound_type_l, bound_type_r, BC_1, BC_2, Lx, delta_x);	
}


void cuda_array::ssti( uint time_step){
	/*
	if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) array_device_freq 	+ time_step*Mx*(Ny/2+1), array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step) % 4)*(Mx+num_gp_x)*(Ny+num_gp_y)
	//if(cufftExecZ2D(plan,(cufftDoubleComplex*) d_x1,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1) )*(Mx+num_gp_x)*(Ny+num_gp_y)
	) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecC2C creation failed\n");	
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	normalize<<<(Ny+num_gp_y)*((Mx+num_gp_x))/64, 64>>>( (time_step)%4, array_device, Mx, Ny, num_gp_x, num_gp_y);
	//normalize<<<(Ny+num_gp_y)*((Mx+num_gp_x))/64, 64>>>( (time_step+1), array_device, Mx, Ny, num_gp_x, num_gp_y);
	

	//BC_array_device<<< (Ny + num_gp_y)*(Mx+ num_gp_x)/64, 64>>>(array_device, time_step, Mx, Ny, num_gp_x, num_gp_y,- 0 ,- 0, Lx);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	if(cufftExecD2Z(plan_f, array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)
	, (cufftDoubleComplex*)array_device_freq 
	+ time_step*Mx*(Ny/2+1)
	) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecC2C creation failed\n");	
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	*/
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	cufftDestroy(plan1);	
///////////////////////////////////////////////////////////////////////////////////////////////////
	rhs<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b, time_step, Ny, Mx, array_arakawa_freq, array_device_freq_deriv, delta_t, 3, delta_x, bound_type_l,bound_type_r, BC_1, BC_2, r);

// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
    //const int nnzA1 = nnzA;
    //const int batchSize1 = batchSize;
    //printf("nnza: %d, batchSize: %d\n", nnzA1, batchSize1);
    cusolver_status = cusolverSpZcsrqrsvBatched(
        cusolverH, m, m, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_b, d_x,
        batchSize,
        info,
        buffer_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	re_arrange_array_freq<<<(Ny*Mx+63)/64 ,64 >>>( d_x, d_x1, time_step, Ny, Mx);

////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaMemcpy(array_device_freq + ((time_step+1) % 4)*(Mx)*(Ny/2+1), d_x1, sizeof(cuDoubleComplex)*Mx*(Ny/2+1), cudaMemcpyDeviceToDevice);
	/*	
	if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) d_x1,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1) % 4)*(Mx+num_gp_x)*(Ny+num_gp_y)
	//if(cufftExecZ2D(plan,(cufftDoubleComplex*) d_x1,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1) )*(Mx+num_gp_x)*(Ny+num_gp_y)
	

	) != CUFFT_SUCCESS){

	fprintf(stderr, "CUFFT error: ExecC2C creation failed\n");	
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	normalize<<<(Ny+num_gp_y)*((Mx+num_gp_x))/64, 64>>>( (time_step+1)%4, array_device, Mx, Ny, num_gp_x, num_gp_y);
	//normalize<<<(Ny+num_gp_y)*((Mx+num_gp_x))/64, 64>>>( (time_step+1), array_device, Mx, Ny, num_gp_x, num_gp_y);
	*/
	////////////////////////////////////////////////////////////////////////////////////////////////////
}


void cuda_array::ssti_without_deriv( uint time_step){

	rhs_without_deriv<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b, time_step, Ny, Mx, array_arakawa_freq, array_device_freq_deriv, delta_t, 3, delta_x, bound_type_l,bound_type_r, BC_1, BC_2, r);

cusolver_status = cusolverSpZcsrqrsvBatched(
        cusolverH, m, m, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_b, d_x,
        batchSize,
        info,
        buffer_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	re_arrange_array_freq<<<(Ny*Mx+63)/64 ,64 >>>( d_x, d_x1, time_step, Ny, Mx);

////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaMemcpy(array_device_freq + ((time_step+1) % 4)*(Mx)*(Ny/2+1), d_x1, sizeof(cuDoubleComplex)*Mx*(Ny/2+1), cudaMemcpyDeviceToDevice);
////////////////////////////////////////////////////////////////////////////////////////////////////
}




void cuda_array::ssti_start( uint time_step, uint order, uint with_deriv){


cusolverSpHandle_t cusolverH7 = NULL;
// GPU does batch QR
    csrqrInfo_t info7 = NULL;
    cusparseMatDescr_t descrA7 = NULL;

    cusparseStatus_t cusparse_status7 = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status7 = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat17 = cudaSuccess;
    cudaError_t cudaStat27 = cudaSuccess;
    cudaError_t cudaStat37 = cudaSuccess;
    cudaError_t cudaStat47 = cudaSuccess;
    cudaError_t cudaStat57 = cudaSuccess;
    cudaError_t cudaStat67 = cudaSuccess;
    int *d_csrRowPtrA7 = NULL;
    int *d_csrColIndA7 = NULL;
    cuDoubleComplex *d_csrValA7 = NULL;
    cuDoubleComplex *d_b7 = NULL; // batchSize * m
    cuDoubleComplex *d_x07 = NULL; // batchSize * m
    cuDoubleComplex *d_x17 = NULL; 

    size_t size_qr7 = 0;
    size_t size_internal7 = 0;
    void *buffer_qr7 = NULL; // working space for numerical factorization

 int m7 = Mx ;
    const int nnzA7 = Mx*3-2;
//    const int csrRowPtrA[m+1]  = { 1, 3, 6,Mx - 1 };

	int *csrRowPtrA7;
	csrRowPtrA7 = (int*)malloc(sizeof(int)*(m7+1));
	csrRowPtrA7[0]=1;
	for( int i = 1; i< m7; i++){
		csrRowPtrA7[i] = 3*i;
	}
	csrRowPtrA7[m7] = nnzA7 +1 ;
 	int *csrColIndA7;
	csrColIndA7 = (int*)malloc(sizeof(int)*(nnzA7));
	csrColIndA7[0]=1;
	csrColIndA7[1]=2;

	for( int i = 1; i< m7-1;i++ ){
		csrColIndA7[3*i-1] =i-1+1;
		csrColIndA7[3*i] = i-1+2;
		csrColIndA7[3*i+1] = i-1+3;
	}
	csrColIndA7[nnzA7-2] = m7-1;
	csrColIndA7[nnzA7-1] = m7;
	//	for(int i = 0; i< nnzA; i++){
//		printf("array at is %d is %d\n", i, csrColIndA[i]);
//	}
	
	//const double csrValA[nnzA] = { 1.0, 1.0, 1.0, 3.0};
    	cuDoubleComplex *csrValA7;
	csrValA7 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA7);
	
	//double dx_2 = (Lx/Mx)*(Lx/Mx);
	//double inv_dx_2 = (Mx/Lx)*(Mx/Lx);

	if(bound_type_l == 0){csrValA7[0]= make_cuDoubleComplex(-3 ,0);}
	else{csrValA7[0]= make_cuDoubleComplex(-1 ,0);}

	csrValA7[1]= make_cuDoubleComplex(1 ,0);

	for( int i = 1; i< m7-1;i++ ){
		csrValA7[3*i-1] = make_cuDoubleComplex(1.0 ,0);
		csrValA7[3*i] = make_cuDoubleComplex(-2.0 ,0);
		csrValA7[3*i+1] = make_cuDoubleComplex(1.0,0);
	}
	csrValA7[nnzA7-2] = make_cuDoubleComplex(1.0 ,0);
	
	if(bound_type_r == 0){csrValA7[nnzA7-1]= make_cuDoubleComplex(-3 ,0);}
	else{csrValA7[nnzA7-1]= make_cuDoubleComplex(-1 ,0);}
	
//	for( int i = 0; i< nnzA; i ++){
//		printf("array at %d is %f\n", i, csrValA[i]);
//	}
  
    //const double b[m] = {1.0, 1.0, 1.0};
    const int batchSize7 = Ny/2 +1;

    cuDoubleComplex *csrValABatch7 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA7*batchSize7);
    cuDoubleComplex *bBatch7       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m7*batchSize7);
    cuDoubleComplex *xBatch7       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m7*batchSize7);
    assert( NULL != csrValABatch7 );
//    assert( NULL != bBatch );
//    assert( NULL != xBatch );

// step 1: prepare Aj and bj on host
//  Aj is a small perturbation of A
//  bj is a small perturbation of b
//  csrValABatch = [A0, A1, A2, ...]
//  bBatch = [b0, b1, b2, ...]
    cuDoubleComplex cons7 = make_cuDoubleComplex((2*M_PI/Ly)*(2*M_PI/Ly),0);
    cuDoubleComplex r7 = make_cuDoubleComplex(diff_coef*delta_t/(delta_x*delta_x), 0);

    double alpha0;	
    if(order==1){ alpha0 = 1.;}
    if(order==2){alpha0 = 1.5;}

	//int counter = 0;
    //double inv_diff_coef = 1/diff_coef;
    for(int colidx7 = 0 ; colidx7 < nnzA7 ; colidx7++){
        cuDoubleComplex Areg7 = csrValA7[colidx7];
        for (int batchId7 = 0 ; batchId7 < batchSize7 ; batchId7++){
            //double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
	    if(Areg7.x != 1){
	    //printf("i: %d k: %f\n", batchId, (double)(batchId%(Ny/2+1)));
	    cuDoubleComplex k7 = make_cuDoubleComplex((batchId7 % (Ny/2+1)),0);
	   // printf("k = %f\n", k.x);
	   //printf("k = %f\n", k.x);
            csrValABatch7[batchId7*nnzA7 + colidx7].x = alpha0+ cons7.x *k7.x*k7.x*delta_t*diff_coef  - Areg7.x*r7.x ;
	    csrValABatch7[batchId7*nnzA7 + colidx7].y = Areg7.y;
		//counter ++;	
	    }
	    else{
	    csrValABatch7[batchId7*nnzA7 + colidx7].x = Areg7.x * (-1.0)*r7.x;
	    csrValABatch7[batchId7*nnzA7 + colidx7].y = Areg7.y;
	}  
	}
    }




// step 2: create cusolver handle, qr info and matrix descriptor
    cusolver_status7 = cusolverSpCreate(&cusolverH7);
    assert (cusolver_status7 == CUSOLVER_STATUS_SUCCESS);

    cusparse_status7 = cusparseCreateMatDescr(&descrA7); 
    assert(cusparse_status7 == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA7, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA7, CUSPARSE_INDEX_BASE_ONE); // base-1

    cusolver_status7 = cusolverSpCreateCsrqrInfo(&info7);
    assert(cusolver_status7 == CUSOLVER_STATUS_SUCCESS);
// step 3: copy Aj and bj to device
    cudaStat17 = cudaMalloc ((void**)&d_csrValA7   , sizeof(cuDoubleComplex) * nnzA7 * batchSize7);
    cudaStat27 = cudaMalloc ((void**)&d_csrColIndA7, sizeof(int) * nnzA7);
    cudaStat37 = cudaMalloc ((void**)&d_csrRowPtrA7, sizeof(int) * (m7+1));
    cudaStat47 = cudaMalloc ((void**)&d_b7         , sizeof(cuDoubleComplex) * m7 * batchSize7);
    cudaStat57 = cudaMalloc ((void**)&d_x07         , sizeof(cuDoubleComplex) * m7 * batchSize7);
    cudaStat67 = cudaMalloc ((void**)&d_x17         , sizeof(cuDoubleComplex) * m7 * batchSize7);
   
    assert(cudaStat17 == cudaSuccess);
    assert(cudaStat27 == cudaSuccess);
    assert(cudaStat37 == cudaSuccess);
    assert(cudaStat47 == cudaSuccess);
    assert(cudaStat57 == cudaSuccess);
    assert(cudaStat67 == cudaSuccess);

    cudaStat17 = cudaMemcpy(d_csrValA7   , csrValABatch7, sizeof(cuDoubleComplex) * nnzA7 * batchSize7, cudaMemcpyHostToDevice);
    cudaStat27 = cudaMemcpy(d_csrColIndA7, csrColIndA7, sizeof(int) * nnzA7, cudaMemcpyHostToDevice);
    cudaStat37 = cudaMemcpy(d_csrRowPtrA7, csrRowPtrA7, sizeof(int) * (m7+1), cudaMemcpyHostToDevice);
//    cudaStat4 = cudaMemcpy(d_b, bBatch, sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);
//	cuDoubleComplex* d_b;
//	d_b = &array_device_freq[ (time_step)*Mx*(Ny/2+1) ];

//	printf("BC_1 in diffusion is %f\n", BC_1);

	//twod_QR_array_freq<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b1, time_step, Ny, Mx, delta_x, bound_type_l,bound_type_r, BC_1, BC_2,r1, delta_t, diff_coef);
 //  }
   
   	if(with_deriv==1){rhs<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b7, time_step, Ny, Mx, array_arakawa_freq, array_device_freq_deriv, delta_t, order, delta_x, bound_type_l,bound_type_r, BC_1, BC_2, r);
	//printf("hallolgalglagalgl\n\n");
	}
	if(with_deriv==0){
	//printf("hallo\n");
	rhs_without_deriv<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b7, time_step, Ny, Mx, array_arakawa_freq, array_device_freq_deriv, delta_t, order, delta_x, bound_type_l,bound_type_r, BC_1, BC_2, r);
	}



    assert(cudaStat17 == cudaSuccess);
    assert(cudaStat27 == cudaSuccess);
    assert(cudaStat37 == cudaSuccess);
//    assert(cudaStat41 == cudaSuccess);


// step 4: symbolic analysis
    cusolver_status7 = cusolverSpXcsrqrAnalysisBatched(
        cusolverH7, m7, m7, nnzA7,
        descrA7, d_csrRowPtrA7, d_csrColIndA7,
        info7);
    assert(cusolver_status7 == CUSOLVER_STATUS_SUCCESS);

// step 5: prepare working space
    cusolver_status7 = cusolverSpZcsrqrBufferInfoBatched(
         cusolverH7, m7, m7, nnzA7,
         descrA7, d_csrValA7, d_csrRowPtrA7, d_csrColIndA7,
         batchSize7,
         info7,
         &size_internal7,
         &size_qr7);
    assert(cusolver_status7 == CUSOLVER_STATUS_SUCCESS);

    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal7);      
    printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr7);      


    cudaStat17 = cudaMalloc((void**)&buffer_qr7, size_qr7);
    assert(cudaStat17 == cudaSuccess);

//	cuDoubleComplex csrValA1 = (cuComplex)csrValA;
	


// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
    cusolver_status7 = cusolverSpZcsrqrsvBatched(
        cusolverH7, m7, m7, nnzA7,
        descrA7, d_csrValA7, d_csrRowPtrA7, d_csrColIndA7,
        d_b7, d_x07,
        batchSize7,
        info7,
        buffer_qr7);
    assert(cusolver_status7 == CUSOLVER_STATUS_SUCCESS);

	re_arrange_array_freq<<<(Ny*Mx+63)/64 ,64 >>>( d_x07, d_x17, time_step, Ny, Mx);

	cudaMemcpy(array_device_freq + ((time_step+1) % 4)*(Mx)*(Ny/2+1), d_x17, sizeof(cuDoubleComplex)*Mx*(Ny/2+1), cudaMemcpyDeviceToDevice);
	
	/*
	if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) d_x11,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1)%4)*(Mx+num_gp_x)*(Ny+num_gp_y)
	//if(cufftExecZ2D(plan,(cufftDoubleComplex*) d_x1,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1))*(Mx+num_gp_x)*(Ny+num_gp_y)
	) != CUFFT_SUCCESS){

	fprintf(stderr, "CUFFT error: ExecC2C creation failed 944\n");	
	}
	
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	normalize<<<((Ny+num_gp_y)*((Mx+num_gp_x))+63)/64, 64>>>( (time_step+1)%4, array_device, Mx, Ny, num_gp_x, num_gp_y);
*/
free(csrRowPtrA7);
free(csrColIndA7);
free(csrValA7);
free(csrValABatch7);
free(bBatch7);
free(xBatch7);

cudaFree(d_csrValA7);
cudaFree(d_csrRowPtrA7);
cudaFree(d_csrColIndA7);
cudaFree(d_x07);
cudaFree(d_x17);
cudaFree(d_b7);
cudaFree(buffer_qr7);
//cufftDestroy(plan_b);











//##########################################################################################################################

/*		
	//if(cufftExecD2Z(plan_f, array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)
	//, (cufftDoubleComplex*)array_device_freq 
	//+ time_step*Mx*(Ny/2+1)
	//) != CUFFT_SUCCESS){
	//fprintf(stderr, "CUFFT error: ExecC2C creation failed 715\n");	
	//}
	
	//if(cudaDeviceSynchronize() != cudaSuccess){
	//fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	//}
	
	//cufftDestroy(plan1);	

cusolverSpHandle_t cusolverH1 = NULL;
// GPU does batch QR
    csrqrInfo_t info1 = NULL;
    cusparseMatDescr_t descrA1 = NULL;

    cusparseStatus_t cusparse_status1 = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status1 = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat11 = cudaSuccess;
    cudaError_t cudaStat21 = cudaSuccess;
    cudaError_t cudaStat31 = cudaSuccess;
    cudaError_t cudaStat41 = cudaSuccess;
    cudaError_t cudaStat51 = cudaSuccess;
    cudaError_t cudaStat61 = cudaSuccess;
    int *d_csrRowPtrA1 = NULL;
    int *d_csrColIndA1 = NULL;
    cuDoubleComplex *d_csrValA1 = NULL;
    cuDoubleComplex *d_b1 = NULL; // batchSize * m
    cuDoubleComplex *d_x01 = NULL; // batchSize * m
    cuDoubleComplex *d_x11 = NULL; 

    size_t size_qr1 = 0;
    size_t size_internal1 = 0;
    void *buffer_qr1 = NULL; // working space for numerical factorization

 int m1 = Mx ;
    const int nnzA1 = Mx*3-2;
//    const int csrRowPtrA[m+1]  = { 1, 3, 6,Mx - 1 };

	int *csrRowPtrA1;
	csrRowPtrA1 = (int*)malloc(sizeof(int)*(m1+1));
	csrRowPtrA1[0]=1;
	for( int i = 1; i< m1; i++){
		csrRowPtrA1[i] = 3*i;
	}
	csrRowPtrA1[m1] = nnzA1 +1 ;
 	int *csrColIndA1;
	csrColIndA1 = (int*)malloc(sizeof(int)*(nnzA1));
	csrColIndA1[0]=1;
	csrColIndA1[1]=2;

	for( int i = 1; i< m1-1;i++ ){
		csrColIndA1[3*i-1] =i-1+1;
		csrColIndA1[3*i] = i-1+2;
		csrColIndA1[3*i+1] = i-1+3;
	}
	csrColIndA1[nnzA1-2] = m1-1;
	csrColIndA1[nnzA1-1] = m1;
	//	for(int i = 0; i< nnzA; i++){
//		printf("array at is %d is %d\n", i, csrColIndA[i]);
//	}
	
	//const double csrValA[nnzA] = { 1.0, 1.0, 1.0, 3.0};
    	cuDoubleComplex *csrValA1;
	csrValA1 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA1);
	
	//double dx_2 = (Lx/Mx)*(Lx/Mx);
	//double inv_dx_2 = (Mx/Lx)*(Mx/Lx);

	if(bound_type_l == 0){csrValA1[0]= make_cuDoubleComplex(-3 ,0);}
	else{csrValA1[0]= make_cuDoubleComplex(-1 ,0);}

	csrValA1[1]= make_cuDoubleComplex(1 ,0);

	for( int i = 1; i< m1-1;i++ ){
		csrValA1[3*i-1] = make_cuDoubleComplex(1.0 ,0);
		csrValA1[3*i] = make_cuDoubleComplex(-2.0 ,0);
		csrValA1[3*i+1] = make_cuDoubleComplex(1.0,0);
	}
	csrValA1[nnzA1-2] = make_cuDoubleComplex(1.0 ,0);
	
	if(bound_type_r == 0){csrValA1[nnzA1-1]= make_cuDoubleComplex(-3 ,0);}
	else{csrValA1[nnzA1-1]= make_cuDoubleComplex(-1 ,0);}
	
//	for( int i = 0; i< nnzA; i ++){
//		printf("array at %d is %f\n", i, csrValA[i]);
//	}
  
    //const double b[m] = {1.0, 1.0, 1.0};
    const int batchSize1 = Ny/2 +1;

    cuDoubleComplex *csrValABatch1 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA1*batchSize1);
    cuDoubleComplex *bBatch1       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m1*batchSize1);
    cuDoubleComplex *xBatch1       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m1*batchSize1);
    assert( NULL != csrValABatch1 );
//    assert( NULL != bBatch );
//    assert( NULL != xBatch );

// step 1: prepare Aj and bj on host
//  Aj is a small perturbation of A
//  bj is a small perturbation of b
//  csrValABatch = [A0, A1, A2, ...]
//  bBatch = [b0, b1, b2, ...]
    cuDoubleComplex cons1 = make_cuDoubleComplex((2*M_PI/Ly)*(2*M_PI/Ly),0);
    cuDoubleComplex r1 = make_cuDoubleComplex(diff_coef*delta_t/(delta_x*delta_x), 0);
	
    double a0;
    if(time_step == 0){a0 = 1.0;}	
    if(time_step == 1){a0 = 3.0/2.0;}
    
	

	//int counter = 0;
    //double inv_diff_coef = 1/diff_coef;
    for(int colidx1 = 0 ; colidx1 < nnzA1 ; colidx1++){
        cuDoubleComplex Areg1 = csrValA1[colidx1];
        for (int batchId1 = 0 ; batchId1 < batchSize1 ; batchId1++){
            //double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
	    if(Areg1.x != 1){
	    //printf("i: %d k: %f\n", batchId, (double)(batchId%(Ny/2+1)));
	    cuDoubleComplex k1 = make_cuDoubleComplex((batchId1 % (Ny/2+1)),0);
	   // printf("k = %f\n", k.x);
	   //printf("k = %f\n", k.x);
            csrValABatch1[batchId1*nnzA1 + colidx1].x = a0+ cons1.x *k1.x*k1.x*delta_t*diff_coef  - Areg1.x*r1.x ;
	    csrValABatch1[batchId1*nnzA1 + colidx1].y = Areg1.y;
		//counter ++;	
	    }
	    else{
	    csrValABatch1[batchId1*nnzA1 + colidx1].x = Areg1.x * (-1.0)*r1.x;
	    csrValABatch1[batchId1*nnzA1 + colidx1].y = Areg1.y;
	}  
	}
    }




// step 2: create cusolver handle, qr info and matrix descriptor
    cusolver_status1 = cusolverSpCreate(&cusolverH1);
    assert (cusolver_status1 == CUSOLVER_STATUS_SUCCESS);

    cusparse_status1 = cusparseCreateMatDescr(&descrA1); 
    assert(cusparse_status1 == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA1, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA1, CUSPARSE_INDEX_BASE_ONE); // base-1

    cusolver_status1 = cusolverSpCreateCsrqrInfo(&info1);
    assert(cusolver_status1 == CUSOLVER_STATUS_SUCCESS);
// step 3: copy Aj and bj to device
    cudaStat11 = cudaMalloc ((void**)&d_csrValA1   , sizeof(cuDoubleComplex) * nnzA1 * batchSize1);
    cudaStat21 = cudaMalloc ((void**)&d_csrColIndA1, sizeof(int) * nnzA1);
    cudaStat31 = cudaMalloc ((void**)&d_csrRowPtrA1, sizeof(int) * (m1+1));
    cudaStat41 = cudaMalloc ((void**)&d_b1         , sizeof(cuDoubleComplex) * m1 * batchSize1);
    cudaStat51 = cudaMalloc ((void**)&d_x01         , sizeof(cuDoubleComplex) * m1 * batchSize1);
    cudaStat61 = cudaMalloc ((void**)&d_x11         , sizeof(cuDoubleComplex) * m1 * batchSize1);
   
    assert(cudaStat11 == cudaSuccess);
    assert(cudaStat21 == cudaSuccess);
    assert(cudaStat31 == cudaSuccess);
    assert(cudaStat41 == cudaSuccess);
    assert(cudaStat51 == cudaSuccess);
    assert(cudaStat61 == cudaSuccess);

    cudaStat11 = cudaMemcpy(d_csrValA1   , csrValABatch1, sizeof(cuDoubleComplex) * nnzA1 * batchSize1, cudaMemcpyHostToDevice);
    cudaStat21 = cudaMemcpy(d_csrColIndA1, csrColIndA1, sizeof(int) * nnzA1, cudaMemcpyHostToDevice);
    cudaStat31 = cudaMemcpy(d_csrRowPtrA1, csrRowPtrA1, sizeof(int) * (m1+1), cudaMemcpyHostToDevice);
//    cudaStat4 = cudaMemcpy(d_b, bBatch, sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);
//	cuDoubleComplex* d_b;
//	d_b = &array_device_freq[ (time_step)*Mx*(Ny/2+1) ];

	
	if(time_step == 0){	
	rhs<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b, time_step, Ny, Mx, array_arakawa_freq, array_device_freq_deriv, delta_t, 1, delta_x, bound_type_l,bound_type_r, BC_1, BC_2, r);
	
	//rhs_1<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b1, time_step, Ny, Mx, array_arakawa_freq, array_device_freq_deriv, delta_t, 1, delta_x, bound_type_l,bound_type_r, BC_1, BC_2, r1);
	}
	else{
	//printf("haajhvkcvjhbkzsjhbakdbh\n");
	rhs<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b, time_step, Ny, Mx, array_arakawa_freq, array_device_freq_deriv, delta_t, 2, delta_x, bound_type_l,bound_type_r, BC_1, BC_2, r);
	}
	
	//twod_QR_array_freq<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b1, time_step, Ny, Mx, delta_x, bound_type_l,bound_type_r, BC_1, BC_2,r1, delta_t, diff_coef);
 //  }
    
    assert(cudaStat11 == cudaSuccess);
    assert(cudaStat21 == cudaSuccess);
    assert(cudaStat31 == cudaSuccess);
//    assert(cudaStat41 == cudaSuccess);


// step 4: symbolic analysis
    cusolver_status1 = cusolverSpXcsrqrAnalysisBatched(
        cusolverH1, m1, m1, nnzA1,
        descrA1, d_csrRowPtrA1, d_csrColIndA1,
        info1);
    assert(cusolver_status1 == CUSOLVER_STATUS_SUCCESS);

// step 5: prepare working space
    cusolver_status1 = cusolverSpZcsrqrBufferInfoBatched(
         cusolverH1, m1, m1, nnzA1,
         descrA1, d_csrValA1, d_csrRowPtrA1, d_csrColIndA1,
         batchSize1,
         info1,
         &size_internal1,
         &size_qr1);
    assert(cusolver_status1 == CUSOLVER_STATUS_SUCCESS);

    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal1);      
    printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr1);      


    cudaStat11 = cudaMalloc((void**)&buffer_qr1, size_qr1);
    assert(cudaStat11 == cudaSuccess);

//	cuDoubleComplex csrValA1 = (cuComplex)csrValA;
	


// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
    cusolver_status1 = cusolverSpZcsrqrsvBatched(
        cusolverH1, m1, m1, nnzA1,
        descrA1, d_csrValA1, d_csrRowPtrA1, d_csrColIndA1,
        d_b1, d_x01,
        batchSize1,
        info1,
        buffer_qr1);
    assert(cusolver_status1 == CUSOLVER_STATUS_SUCCESS);

	re_arrange_array_freq<<<(Ny*Mx+63)/64 ,64 >>>( d_x01, d_x11, time_step, Ny, Mx);

	cudaMemcpy(array_device_freq + ((time_step+1) % 4)*(Mx)*(Ny/2+1), d_x11, sizeof(cuDoubleComplex)*Mx*(Ny/2+1), cudaMemcpyDeviceToDevice);
	
	
	//if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) d_x11,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1)%4)*(Mx+num_gp_x)*(Ny+num_gp_y)
	//if(cufftExecZ2D(plan,(cufftDoubleComplex*) d_x1,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1))*(Mx+num_gp_x)*(Ny+num_gp_y)
	//) != CUFFT_SUCCESS){

	//fprintf(stderr, "CUFFT error: ExecC2C creation failed 944\n");	
	//}
	
	//if(cudaDeviceSynchronize() != cudaSuccess){
	//fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	//}
	
	//normalize<<<((Ny+num_gp_y)*((Mx+num_gp_x))+63)/64, 64>>>( (time_step+1)%4, array_device, Mx, Ny, num_gp_x, num_gp_y);

free(csrRowPtrA1);
free(csrColIndA1);
free(csrValA1);
free(csrValABatch1);
free(bBatch1);
free(xBatch1);

cudaFree(d_csrValA1);
cudaFree(d_csrRowPtrA1);
cudaFree(d_csrColIndA1);
cudaFree(d_x01);
cudaFree(d_x11);
cudaFree(d_b1);
cudaFree(buffer_qr1);

*/
}


void cuda_array::diffusion_2d( uint time_step){
	/*
	if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) array_device_freq 	+ time_step*Mx*(Ny/2+1), array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step) % 4)*(Mx+num_gp_x)*(Ny+num_gp_y)
	//if(cufftExecZ2D(plan,(cufftDoubleComplex*) d_x1,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1) )*(Mx+num_gp_x)*(Ny+num_gp_y)
	) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecC2C creation failed\n");	
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	normalize<<<(Ny+num_gp_y)*((Mx+num_gp_x))/64+63, 64>>>( (time_step)%4, array_device, Mx, Ny, num_gp_x, num_gp_y);
	//normalize<<<(Ny+num_gp_y)*((Mx+num_gp_x))/64, 64>>>( (time_step+1), array_device, Mx, Ny, num_gp_x, num_gp_y);
	
	 cuDoubleComplex r1 = make_cuDoubleComplex(diff_coef*delta_t/(delta_x*delta_x), 0);
	//printf("BC1 =  %f\n", BC_1);

	gp_device<<<(Ny+num_gp_y)*((Mx+num_gp_x))/64+63, 64>>>(array_device,time_step,Mx,Ny,num_gp_x,num_gp_y,bound_type_l,bound_type_r,BC_1,BC_2,Lx,delta_x, r1);
	
	//BC_array_device<<< (Ny + num_gp_y)*(Mx+ num_gp_x)/64, 64>>>(array_device, time_step, Mx, Ny, num_gp_x, num_gp_y,- 0 ,- 0, Lx);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	if(cufftExecD2Z(plan_f, array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)
	, (cufftDoubleComplex*)array_device_freq 
	+ time_step*Mx*(Ny/2+1)
	) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecC2C creation failed\n");	
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	*/



cusolverSpHandle_t cusolverH1 = NULL;
// GPU does batch QR
    csrqrInfo_t info1 = NULL;
    cusparseMatDescr_t descrA1 = NULL;

    cusparseStatus_t cusparse_status1 = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status1 = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat11 = cudaSuccess;
    cudaError_t cudaStat21 = cudaSuccess;
    cudaError_t cudaStat31 = cudaSuccess;
    cudaError_t cudaStat41 = cudaSuccess;
    cudaError_t cudaStat51 = cudaSuccess;
    cudaError_t cudaStat61 = cudaSuccess;
    int *d_csrRowPtrA1 = NULL;
    int *d_csrColIndA1 = NULL;
    cuDoubleComplex *d_csrValA1 = NULL;
    cuDoubleComplex *d_b1 = NULL; // batchSize * m
    cuDoubleComplex *d_x01 = NULL; // batchSize * m
    cuDoubleComplex *d_x11 = NULL; 

    size_t size_qr1 = 0;
    size_t size_internal1 = 0;
    void *buffer_qr1 = NULL; // working space for numerical factorization

 int m1 = Mx ;
    const int nnzA1 = Mx*3-2;
//    const int csrRowPtrA[m+1]  = { 1, 3, 6,Mx - 1 };

	int *csrRowPtrA1;
	csrRowPtrA1 = (int*)malloc(sizeof(int)*(m1+1));
	csrRowPtrA1[0]=1;
	for( int i = 1; i< m1; i++){
		csrRowPtrA1[i] = 3*i;
	}
	csrRowPtrA1[m1] = nnzA1 +1 ;
 	int *csrColIndA1;
	csrColIndA1 = (int*)malloc(sizeof(int)*(nnzA1));
	csrColIndA1[0]=1;
	csrColIndA1[1]=2;

	for( int i = 1; i< m1-1;i++ ){
		csrColIndA1[3*i-1] =i-1+1;
		csrColIndA1[3*i] = i-1+2;
		csrColIndA1[3*i+1] = i-1+3;
	}
	csrColIndA1[nnzA1-2] = m1-1;
	csrColIndA1[nnzA1-1] = m1;
	//	for(int i = 0; i< nnzA; i++){
//		printf("array at is %d is %d\n", i, csrColIndA[i]);
//	}
	
	//const double csrValA[nnzA] = { 1.0, 1.0, 1.0, 3.0};
    	cuDoubleComplex *csrValA1;
	csrValA1 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA1);
	
	//double dx_2 = (Lx/Mx)*(Lx/Mx);
	//double inv_dx_2 = (Mx/Lx)*(Mx/Lx);

	if(bound_type_l == 0){csrValA1[0]= make_cuDoubleComplex(-3 ,0);}
	else{csrValA1[0]= make_cuDoubleComplex(-1 ,0);}

	csrValA1[1]= make_cuDoubleComplex(1 ,0);

	for( int i = 1; i< m1-1;i++ ){
		csrValA1[3*i-1] = make_cuDoubleComplex(1.0 ,0);
		csrValA1[3*i] = make_cuDoubleComplex(-2.0 ,0);
		csrValA1[3*i+1] = make_cuDoubleComplex(1.0,0);
	}
	csrValA1[nnzA1-2] = make_cuDoubleComplex(1.0 ,0);
	
	if(bound_type_r == 0){csrValA1[nnzA1-1]= make_cuDoubleComplex(-3 ,0);}
	else{csrValA1[nnzA1-1]= make_cuDoubleComplex(-1 ,0);}
	
//	for( int i = 0; i< nnzA; i ++){
//		printf("array at %d is %f\n", i, csrValA[i]);
//	}
  
    //const double b[m] = {1.0, 1.0, 1.0};
    const int batchSize1 = Ny/2 +1;

    cuDoubleComplex *csrValABatch1 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*nnzA1*batchSize1);
    cuDoubleComplex *bBatch1       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m1*batchSize1);
    cuDoubleComplex *xBatch1       = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex)*m1*batchSize1);
    assert( NULL != csrValABatch1 );
//    assert( NULL != bBatch );
//    assert( NULL != xBatch );

// step 1: prepare Aj and bj on host
//  Aj is a small perturbation of A
//  bj is a small perturbation of b
//  csrValABatch = [A0, A1, A2, ...]
//  bBatch = [b0, b1, b2, ...]
    cuDoubleComplex cons1 = make_cuDoubleComplex((2*M_PI/Ly)*(2*M_PI/Ly),0);
    cuDoubleComplex r1 = make_cuDoubleComplex(diff_coef*delta_t/(delta_x*delta_x), 0);

	//int counter = 0;
    //double inv_diff_coef = 1/diff_coef;
    for(int colidx1 = 0 ; colidx1 < nnzA1 ; colidx1++){
        cuDoubleComplex Areg1 = csrValA1[colidx1];
        for (int batchId1 = 0 ; batchId1 < batchSize1 ; batchId1++){
            //double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
	    if(Areg1.x != 1){
	    //printf("i: %d k: %f\n", batchId, (double)(batchId%(Ny/2+1)));
	    cuDoubleComplex k1 = make_cuDoubleComplex((batchId1 % (Ny/2+1)),0);
	   // printf("k = %f\n", k.x);
	   //printf("k = %f\n", k.x);
            csrValABatch1[batchId1*nnzA1 + colidx1].x = 1+ cons1.x *k1.x*k1.x*delta_t*diff_coef  - Areg1.x*r1.x ;
	    csrValABatch1[batchId1*nnzA1 + colidx1].y = Areg1.y;
		//counter ++;	
	    }
	    else{
	    csrValABatch1[batchId1*nnzA1 + colidx1].x = Areg1.x * (-1.0)*r1.x;
	    csrValABatch1[batchId1*nnzA1 + colidx1].y = Areg1.y;
	}  
	}
    }




// step 2: create cusolver handle, qr info and matrix descriptor
    cusolver_status1 = cusolverSpCreate(&cusolverH1);
    assert (cusolver_status1 == CUSOLVER_STATUS_SUCCESS);

    cusparse_status1 = cusparseCreateMatDescr(&descrA1); 
    assert(cusparse_status1 == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA1, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA1, CUSPARSE_INDEX_BASE_ONE); // base-1

    cusolver_status1 = cusolverSpCreateCsrqrInfo(&info1);
    assert(cusolver_status1 == CUSOLVER_STATUS_SUCCESS);
// step 3: copy Aj and bj to device
    cudaStat11 = cudaMalloc ((void**)&d_csrValA1   , sizeof(cuDoubleComplex) * nnzA1 * batchSize1);
    cudaStat21 = cudaMalloc ((void**)&d_csrColIndA1, sizeof(int) * nnzA1);
    cudaStat31 = cudaMalloc ((void**)&d_csrRowPtrA1, sizeof(int) * (m1+1));
    cudaStat41 = cudaMalloc ((void**)&d_b1         , sizeof(cuDoubleComplex) * m1 * batchSize1);
    cudaStat51 = cudaMalloc ((void**)&d_x01         , sizeof(cuDoubleComplex) * m1 * batchSize1);
    cudaStat61 = cudaMalloc ((void**)&d_x11         , sizeof(cuDoubleComplex) * m1 * batchSize1);
   
    assert(cudaStat11 == cudaSuccess);
    assert(cudaStat21 == cudaSuccess);
    assert(cudaStat31 == cudaSuccess);
    assert(cudaStat41 == cudaSuccess);
    assert(cudaStat51 == cudaSuccess);
    assert(cudaStat61 == cudaSuccess);

    cudaStat11 = cudaMemcpy(d_csrValA1   , csrValABatch1, sizeof(cuDoubleComplex) * nnzA1 * batchSize1, cudaMemcpyHostToDevice);
    cudaStat21 = cudaMemcpy(d_csrColIndA1, csrColIndA1, sizeof(int) * nnzA1, cudaMemcpyHostToDevice);
    cudaStat31 = cudaMemcpy(d_csrRowPtrA1, csrRowPtrA1, sizeof(int) * (m1+1), cudaMemcpyHostToDevice);
//    cudaStat4 = cudaMemcpy(d_b, bBatch, sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);
//	cuDoubleComplex* d_b;
//	d_b = &array_device_freq[ (time_step)*Mx*(Ny/2+1) ];

//	printf("BC_1 in diffusion is %f\n", BC_1);

	twod_QR_array_freq<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq ,d_b1, time_step, Ny, Mx, delta_x, bound_type_l,bound_type_r, BC_1, BC_2,r1, delta_t, diff_coef);
 //  }
    
    assert(cudaStat11 == cudaSuccess);
    assert(cudaStat21 == cudaSuccess);
    assert(cudaStat31 == cudaSuccess);
//    assert(cudaStat41 == cudaSuccess);


// step 4: symbolic analysis
    cusolver_status1 = cusolverSpXcsrqrAnalysisBatched(
        cusolverH1, m1, m1, nnzA1,
        descrA1, d_csrRowPtrA1, d_csrColIndA1,
        info1);
    assert(cusolver_status1 == CUSOLVER_STATUS_SUCCESS);

// step 5: prepare working space
    cusolver_status1 = cusolverSpZcsrqrBufferInfoBatched(
         cusolverH1, m1, m1, nnzA1,
         descrA1, d_csrValA1, d_csrRowPtrA1, d_csrColIndA1,
         batchSize1,
         info1,
         &size_internal1,
         &size_qr1);
    assert(cusolver_status1 == CUSOLVER_STATUS_SUCCESS);

    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal1);      
    printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr1);      


    cudaStat11 = cudaMalloc((void**)&buffer_qr1, size_qr1);
    assert(cudaStat11 == cudaSuccess);

//	cuDoubleComplex csrValA1 = (cuComplex)csrValA;
	


// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
    cusolver_status1 = cusolverSpZcsrqrsvBatched(
        cusolverH1, m1, m1, nnzA1,
        descrA1, d_csrValA1, d_csrRowPtrA1, d_csrColIndA1,
        d_b1, d_x01,
        batchSize1,
        info1,
        buffer_qr1);
    assert(cusolver_status1 == CUSOLVER_STATUS_SUCCESS);

	re_arrange_array_freq<<<(Ny*Mx+63)/64 ,64 >>>( d_x01, d_x11, time_step, Ny, Mx);

	cudaMemcpy(array_device_freq + ((time_step+1) % 4)*(Mx)*(Ny/2+1), d_x11, sizeof(cuDoubleComplex)*Mx*(Ny/2+1), cudaMemcpyDeviceToDevice);
	
	/*
	if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) d_x11,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1)%4)*(Mx+num_gp_x)*(Ny+num_gp_y)
	//if(cufftExecZ2D(plan,(cufftDoubleComplex*) d_x1,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + ((time_step+1))*(Mx+num_gp_x)*(Ny+num_gp_y)
	) != CUFFT_SUCCESS){

	fprintf(stderr, "CUFFT error: ExecC2C creation failed 944\n");	
	}
	
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	normalize<<<((Ny+num_gp_y)*((Mx+num_gp_x))+63)/64, 64>>>( (time_step+1)%4, array_device, Mx, Ny, num_gp_x, num_gp_y);
*/
free(csrRowPtrA1);
free(csrColIndA1);
free(csrValA1);
free(csrValABatch1);
free(bBatch1);
free(xBatch1);

cudaFree(d_csrValA1);
cudaFree(d_csrRowPtrA1);
cudaFree(d_csrColIndA1);
cudaFree(d_x01);
cudaFree(d_x11);
cudaFree(d_b1);
cudaFree(buffer_qr1);
//cufftDestroy(plan_b);
}


void cuda_array::diffusion_x( uint time_step){

cusolverSpHandle_t cusolverH = NULL;
// GPU does batch QR
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;


// GPU does batch QR
// d_A is CSR format, d_csrValA is of size nnzA*batchSize
// d_x is a matrix of size batchSize * m
// d_b is a matrix of size batchSize * m
    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;
    double *d_b = NULL; // batchSize * m
    double *d_x = NULL; // batchSize * m
	double *d_x1 = NULL;

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = NULL; // working space for numerical factorization
 int m = Mx ;
    const int nnzA = Mx*3-2;
//    const int csrRowPtrA[m+1]  = { 1, 3, 6,Mx - 1 };

	int *csrRowPtrA;
	csrRowPtrA = (int*)malloc(sizeof(int)*(m+1));
	csrRowPtrA[0]=1;
	for( int i = 1; i< m; i++){
		csrRowPtrA[i] = 3*i;
	}
	csrRowPtrA[m] = nnzA +1 ;
 	int *csrColIndA;
	csrColIndA = (int*)malloc(sizeof(int)*(nnzA));
	csrColIndA[0]=1;
	csrColIndA[1]=2;

	for( int i = 1; i< m-1;i++ ){
		csrColIndA[3*i-1] =i-1+1;
		csrColIndA[3*i] = i-1+2;
		csrColIndA[3*i+1] = i-1+3;
	}
	csrColIndA[nnzA-2] = m-1;
	csrColIndA[nnzA-1] = m;
    	double *csrValA;
	csrValA = (double*)malloc(sizeof(double)*nnzA);
	
	double r = delta_t/(delta_x*delta_x);
	
	csrValA[0]=1+3*r;
	csrValA[1]=-r;

	for( int i = 1; i< m-1;i++ ){
		csrValA[3*i-1] =-r;
		csrValA[3*i] = 1+2*r;
		csrValA[3*i+1] = -r;
	}
	csrValA[nnzA-2] = -r;
	csrValA[nnzA-1] = 1+3*r;
    const int batchSize = Ny;

    double *csrValABatch = (double*)malloc(sizeof(double)*nnzA*batchSize);
    double *bBatch       = (double*)malloc(sizeof(double)*m*batchSize);
    double *xBatch       = (double*)malloc(sizeof(double)*m*batchSize);
    assert( NULL != csrValABatch );
    assert( NULL != bBatch );
    assert( NULL != xBatch );

// step 1: prepare Aj and bj on host
//  Aj is a small perturbation of A
//  bj is a small perturbation of b
//  csrValABatch = [A0, A1, A2, ...]
//  bBatch = [b0, b1, b2, ...]
    for(int colidx = 0 ; colidx < nnzA ; colidx++){
        double Areg = csrValA[colidx];
        for (int batchId = 0 ; batchId < batchSize ; batchId++){
            //double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
            csrValABatch[batchId*nnzA + colidx] = Areg;// + eps;
        }  
    }

// step 2: create cusolver handle, qr info and matrix descriptor
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusparse_status = cusparseCreateMatDescr(&descrA); 
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); // base-1

    cusolver_status = cusolverSpCreateCsrqrInfo(&info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
// step 3: copy Aj and bj to device
    cudaStat1 = cudaMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA * batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA);
    cudaStat3 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (m+1));
    cudaStat4 = cudaMalloc ((void**)&d_b         , sizeof(double) * m * batchSize);
    cudaStat5 = cudaMalloc ((void**)&d_x         , sizeof(double) * (Ny+num_gp_y)*(Mx+num_gp_x));
    cudaStat6 = cudaMalloc ((void**)&d_x1         , sizeof(double) * (Ny+num_gp_y)*(Mx+num_gp_x));
    assert(cudaStat6 == cudaSuccess);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);

    cudaStat1 = cudaMemcpy(d_csrValA   , csrValABatch, sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m+1), cudaMemcpyHostToDevice);
    //cudaStat4 = cudaMemcpy(d_b, bBatch, sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    //assert(cudaStat4 == cudaSuccess);


time_int_reorder<<< ((Mx+num_gp_x)*(Ny+num_gp_y)+31)/32,32 >>>(array_device, d_b, time_step, Ny, Mx , num_gp_x, num_gp_y);
	

// step 4: symbolic analysis
    cusolver_status = cusolverSpXcsrqrAnalysisBatched(
        cusolverH, m, m, nnzA,
        descrA, d_csrRowPtrA, d_csrColIndA,
        info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

// step 5: prepare working space
    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
         cusolverH, m, m, nnzA,
         descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
         batchSize,
         info,
         &size_internal,
         &size_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);      
    printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr);      


    cudaStat1 = cudaMalloc((void**)&buffer_qr, size_qr);
    assert(cudaStat1 == cudaSuccess);

// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
    cusolver_status = cusolverSpDcsrqrsvBatched(
        cusolverH, m, m, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_b, d_x,
        batchSize,
        info,
        buffer_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	re_arrange_array_real<<<  ( (Mx+num_gp_x)*(Ny+num_gp_y)+31)/32,32>>>(d_x, d_x1, time_step, Ny, Mx, num_gp_x, num_gp_y); 

//gpuErrchk(cudaMemcpy(array_device + (Ny+num_gp_y)*(Mx+num_gp_x)*((time_step +1)%4 ), d_x1, sizeof(double)*(Ny+num_gp_y)*(Mx+num_gp_x), cudaMemcpyDeviceToDevice));
gpuErrchk(cudaMemcpy(array_device + (Ny+num_gp_y)*(Mx+num_gp_x)*((time_step +1) ), d_x1, sizeof(double)*(Ny+num_gp_y)*(Mx+num_gp_x), cudaMemcpyDeviceToDevice));


//delete b;
free(csrRowPtrA);
free(csrColIndA);
free(csrValA);
free(csrValABatch);
free(bBatch);
free(xBatch);

cudaFree(d_csrValA);
cudaFree(d_csrRowPtrA);
cudaFree(d_csrColIndA);
cudaFree(d_x);
cudaFree(d_b);
cudaFree(d_x1);
cudaFree(buffer_qr);
}



void cuda_array::QR_factorisation_2d( uint time_step, CuCmplx<double>* array){
//printf("csrRowPtrA_QR adress anfang QR= %p\n", csrRowPtrA_QR);
	

	//BC_array_device<<< (Ny + num_gp_y)*(Mx+ num_gp_x)/64, 64>>>(array_device, time_step, Mx, Ny, num_gp_x, num_gp_y,- 0 ,- 0, Lx);
/*			
	if(cufftExecD2Z(plan_f, array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)
	, (cufftDoubleComplex*)array_device_freq + time_step*Mx*(Ny/2+1)) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecC2C creation failed\n");	
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
*/	
	twod_QR_array_freq<<<((Ny*Mx+63)/64) ,64 >>>(array,d_b, time_step, Ny, Mx, delta_x, bound_type_l, bound_type_r, BC_1, BC_2,r, delta_t, diff_coef);
	//twod_QR_array_freq<<<((Ny*Mx+63)/64) ,64 >>>(array_device_freq,d_b, time_step, Ny, Mx, delta_x, bound_type_l, bound_type_r, BC_1, BC_2,r, delta_t, diff_coef);
 



// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
    cusolver_status_QR = cusolverSpZcsrqrsvBatched(
        cusolverH, Mx, Mx, nnzA,
        descrA, QR_values, d_csrRowPtrA, d_csrColIndA,
        d_b, d_x,
        Ny/2+1,
        info,
        buffer_qr);
    assert(cusolver_status_QR == CUSOLVER_STATUS_SUCCESS);

	re_arrange_array_freq<<<(Ny*Mx+63)/64 ,64 >>>( d_x, d_x1, time_step, Ny, Mx);
	
	
	cudaMemcpy(array_device_freq + ((time_step) )*(Mx)*(Ny/2+1), d_x1, sizeof(cuDoubleComplex)*Mx*(Ny/2+1), cudaMemcpyDeviceToDevice);
	//printf("hallo\n\n");	

/*	
	if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) array_device_freq + ((time_step) % 4)*(Mx)*(Ny/2+1),  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)
	) != CUFFT_SUCCESS){

	fprintf(stderr, "CUFFT error: ExecC2C creation failed");	
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	normalize<<<(Ny+num_gp_y)*((Mx+num_gp_x))/64, 64>>>( time_step, array_device, Mx, Ny, num_gp_x, num_gp_y);
*/	
}



void cuda_array::QR_factorisation_1d( uint time_step){

double dx = (Lx/(Mx));
cusolverSpHandle_t cusolverH = NULL;
// GPU does batch QR
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;


// GPU does batch QR
// d_A is CSR format, d_csrValA is of size nnzA*batchSize
// d_x is a matrix of size batchSize * m
// d_b is a matrix of size batchSize * m
    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;
    double *d_b = NULL; // batchSize * m
    double *d_x = NULL; // batchSize * m
	double *d_x1 = NULL;

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = NULL; // working space for numerical factorization

 int m = Mx ;
    const int nnzA = Mx*3-2;
//    const int csrRowPtrA[m+1]  = { 1, 3, 6,Mx - 1 };

	int *csrRowPtrA;
	csrRowPtrA = (int*)malloc(sizeof(int)*(m+1));
	csrRowPtrA[0]=1;
	for( int i = 1; i< m; i++){
		csrRowPtrA[i] = 3*i;
	}
	csrRowPtrA[m] = nnzA +1 ;

 	int *csrColIndA;
	csrColIndA = (int*)malloc(sizeof(int)*(nnzA));
	csrColIndA[0]=1;
	csrColIndA[1]=2;

	for( int i = 1; i< m-1;i++ ){
		csrColIndA[3*i-1] =i-1+1;
		csrColIndA[3*i] = i-1+2;
		csrColIndA[3*i+1] = i-1+3;
	}
	csrColIndA[nnzA-2] = m-1;
	csrColIndA[nnzA-1] = m;
	
    	double *csrValA;
	csrValA = (double*)malloc(sizeof(double)*nnzA);
	csrValA[0]=-3;
	csrValA[1]=1;

	for( int i = 1; i< m-1;i++ ){
		csrValA[3*i-1] =1;
		csrValA[3*i] = -2;
		csrValA[3*i+1] = 1;
	}
	csrValA[nnzA-2] = 1;
	csrValA[nnzA-1] = -3;
    const int batchSize = Ny;

    double *csrValABatch = (double*)malloc(sizeof(double)*nnzA*batchSize);
    double *bBatch       = (double*)malloc(sizeof(double)*m*batchSize);
    double *xBatch       = (double*)malloc(sizeof(double)*m*batchSize);
    assert( NULL != csrValABatch );
    assert( NULL != bBatch );
    assert( NULL != xBatch );

// step 1: prepare Aj and bj on host
//  Aj is a small perturbation of A
//  bj is a small perturbation of b
//  csrValABatch = [A0, A1, A2, ...]
//  bBatch = [b0, b1, b2, ...]
    for(int colidx = 0 ; colidx < nnzA ; colidx++){
        double Areg = csrValA[colidx];
        for (int batchId = 0 ; batchId < batchSize ; batchId++){
            //double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
            csrValABatch[batchId*nnzA + colidx] = Areg;// + eps;
        }  
    }

// step 2: create cusolver handle, qr info and matrix descriptor
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusparse_status = cusparseCreateMatDescr(&descrA); 
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); // base-1

    cusolver_status = cusolverSpCreateCsrqrInfo(&info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
// step 3: copy Aj and bj to device
    cudaStat1 = cudaMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA * batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA);
    cudaStat3 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (m+1));
    cudaStat4 = cudaMalloc ((void**)&d_b         , sizeof(double) * m * batchSize);
    cudaStat5 = cudaMalloc ((void**)&d_x         , sizeof(double) * (Ny+num_gp_y)*(Mx+num_gp_x));
    cudaStat6 = cudaMalloc ((void**)&d_x1         , sizeof(double) * (Ny+num_gp_y)*(Mx+num_gp_x));
    assert(cudaStat6 == cudaSuccess);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);

    cudaStat1 = cudaMemcpy(d_csrValA   , csrValABatch, sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m+1), cudaMemcpyHostToDevice);
    //cudaStat4 = cudaMemcpy(d_b, bBatch, sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    //assert(cudaStat4 == cudaSuccess);


twod_QR_array_real<<<( (Mx+num_gp_x)*(Ny+num_gp_y)+31)/32,32 >>>(array_device, d_b, time_step, Ny, Mx , num_gp_x, num_gp_y, dx);
	

// step 4: symbolic analysis
    cusolver_status = cusolverSpXcsrqrAnalysisBatched(
        cusolverH, m, m, nnzA,
        descrA, d_csrRowPtrA, d_csrColIndA,
        info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

// step 5: prepare working space
    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
         cusolverH, m, m, nnzA,
         descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
         batchSize,
         info,
         &size_internal,
         &size_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);      
    printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr);      


    cudaStat1 = cudaMalloc((void**)&buffer_qr, size_qr);
    assert(cudaStat1 == cudaSuccess);

// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
    cusolver_status = cusolverSpDcsrqrsvBatched(
        cusolverH, m, m, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_b, d_x,
        batchSize,
        info,
        buffer_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	re_arrange_array_real<<<  ( (Mx+num_gp_x)*(Ny+num_gp_y)+31)/32,32>>>(d_x, d_x1, time_step, Ny, Mx, num_gp_x, num_gp_y); 

gpuErrchk(cudaMemcpy(array_device + (Ny+num_gp_y)*(Mx+num_gp_x)*(time_step), d_x1, sizeof(double)*(Ny+num_gp_y)*(Mx+num_gp_x), cudaMemcpyDeviceToDevice));



//delete b;
free(csrRowPtrA);
free(csrColIndA);
free(csrValA);
free(csrValABatch);
free(bBatch);
free(xBatch);

cudaFree(d_csrValA);
cudaFree(d_csrRowPtrA);
cudaFree(d_csrColIndA);
cudaFree(d_x);
cudaFree(d_x1);
cudaFree(d_b);
cudaFree(buffer_qr);

}


int cuda_array::adress(uint time_step, uint Mx1, uint Ny1){
	int offset = ((time_step)*(Mx+num_gp_x)*(Ny+num_gp_y)+(Ny + num_gp_y)*(Mx1+1) + num_gp_y/2 + Ny1);
	return offset;
}

void cuda_array::dirichlet( uint time_step, double value1, double value2){
	dirichlet_cuda<<<(Ny+num_gp_y)*(Mx+num_gp_x)/4, 4>>>(time_step, array_device, Mx, Ny, num_gp_x, num_gp_y, size, value1, value2);
}

void cuda_array::neumann( uint time_step, double value1, double value2){
	neumann_cuda<<<(Ny+num_gp_y)*(Mx+num_gp_x)/4, 4>>>(time_step, array_device_deriv, Mx, Ny, num_gp_x, num_gp_y, size, value1, value2, Lx);
}

void cuda_array::clean_all( uint time__step){
	//int i;
	//for(i = 0; i < size; i++){
	//	array_host[i] = 0;
	//}
	clean_all_cuda<<<((Ny+num_gp_y)*(Mx+num_gp_x)+63)/64, 64>>>(array_device,array_device_deriv, size, time_step, Ny, num_gp_y, Mx, num_gp_x);
}

void cuda_array::alloc_host_real(){
//	double* array_host;
	array_host = new double[(Mx+num_gp_x)*(Ny+num_gp_y)]; 
//	array_host2 = new double[size];
//	array_host = (double*)malloc(sizeof(double)*size);
}

void cuda_array::alloc_device_real(){
	gpuErrchk(cudaMalloc((void**) &array_device, sizeof(cufftDoubleReal)*(Mx+num_gp_x)*(Ny+num_gp_y)*4));
}

void cuda_array::alloc_device_deriv(){
	gpuErrchk(cudaMalloc((void**) &array_device_deriv, sizeof(cufftDoubleReal)*(Mx+num_gp_x)*(Ny+num_gp_y)*4));
}

void cuda_array::alloc_device_freq(){
	gpuErrchk(cudaMalloc((void**) &array_device_freq, sizeof(CuCmplx<double>)*Mx*(Ny/2+1)*4));
}

void cuda_array::alloc_device_freq_deriv(){
	gpuErrchk(cudaMalloc((void**) &array_device_freq_deriv, sizeof(CuCmplx<double>)*Mx*(Ny/2+1)*4));
}

void cuda_array::alloc_arakawa(){
	gpuErrchk(cudaMalloc((void**) &array_arakawa, sizeof(double)*(Mx+num_gp_y)*(Ny+num_gp_y)*4));
}

void cuda_array::alloc_arakawa_freq(){
	gpuErrchk(cudaMalloc((void**) &array_arakawa_freq, sizeof(CuCmplx<double>)*Mx*(Ny/2+1)*4));
}

void cuda_array::copy_host_to_device( uint time_step){
	gpuErrchk(cudaMemcpy(array_device+ (Mx+num_gp_x)*(Ny+num_gp_y)*(time_step%4), array_host , sizeof(double)*(Mx+num_gp_x)*(Ny+num_gp_y), cudaMemcpyHostToDevice));
}

void cuda_array::copy_device_to_host( uint time_step){
	gpuErrchk(cudaMemcpy(array_host, array_device + (Mx+num_gp_x)*(Ny+num_gp_y)*(time_step%4), sizeof(double)*(Mx+num_gp_x)*(Ny+num_gp_y), cudaMemcpyDeviceToHost));
}

void cuda_array::copy_arakawa_to_host( uint time_step){
	gpuErrchk(cudaMemcpy(array_host, array_arakawa + (Mx+num_gp_x)*(Ny+num_gp_y)*(time_step%4), sizeof(double)*(Mx+num_gp_x)*(Ny+num_gp_y), cudaMemcpyDeviceToHost));
}

void cuda_array::copy_device_deriv_to_host( uint time_step){
	gpuErrchk(cudaMemcpy(array_host, array_device_deriv + (Mx+num_gp_x)*(Ny+num_gp_y)*(time_step%4), sizeof(double)*(Mx+num_gp_x)*(Ny+num_gp_y), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(array_host, array_device_deriv, sizeof(double)*size, cudaMemcpyDeviceToHost));
	/*
	int i;
	cufftDoubleReal *out3;
	out3 = (cufftDoubleReal*) malloc(sizeof(cufftDoubleReal) *size);
	gpuErrchk(cudaMemcpy(out3, array_device_deriv, sizeof(double)*size , cudaMemcpyDeviceToHost));
	printf("\n\n\n\n\n");

	for(i = 0; i <size; i++){
		printf("%d: %f\n", i, out3[i]);
	}
	free(out3);
	*/
}

void cuda_array::deriv_real_y( uint time_step, cufftDoubleReal* array){
	deriv_real_y_cuda<<<((Ny+num_gp_y)*(Mx+num_gp_x)+63)/64,64>>>( array, array_device_deriv, time_step, Ny, Ly, num_gp_x, num_gp_y , size, Mx );
	
				
	if(cufftExecD2Z(plan_f, array_device_deriv +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)
	, (cufftDoubleComplex*)array_device_freq_deriv + time_step*Mx*(Ny/2+1)) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecC2C creation failed\n");	
	}
	
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	
	/*
	//remove next lines in final Programm
	int i;
	cufftDoubleReal *out;
	out = (cufftDoubleReal*) malloc(sizeof(cufftDoubleReal) *size);
	gpuErrchk(cudaMemcpy(out, array_device, sizeof(cufftDoubleReal)*size , cudaMemcpyDeviceToHost));
	
	printf("\n\n\n\n\n");

	for(i = 0; i <size; i++){
		printf("%d: %f\n", i, out[i]);
	}
	free(out);
	*/
}

void cuda_array::deriv_freq_y( uint time_step, CuCmplx<double>*  array){
	deriv_freq_y_cuda<<<(Mx*(Ny/2+1)+63)/64,64>>>( array, array_device_freq_deriv, time_step, Ny, Mx, Ly);
	/*	
	if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) array_device_freq_deriv + time_step*(Ny/2+1)*Mx,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)
	) != CUFFT_SUCCESS){

	fprintf(stderr, "CUFFT error: ExecC2C creation failed");	
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	normalize<<<(Ny+num_gp_y)*((Mx+num_gp_x))/64, 64>>>( time_step, array_device, Mx, Ny, num_gp_x, num_gp_y);
	*/	
}

void cuda_array::second_deriv_freq_y( uint time_step){
	second_deriv_freq_y_cuda<<<((Ny+num_gp_y)*(Mx+num_gp_x)+63)/64,64>>>( array_device_freq, array_device_freq_deriv, time_step, Ny, Mx, Ly);
}


void cuda_array::second_deriv_real_y( uint time_step){
	second_deriv_real_y_cuda<<<((Ny+num_gp_y)*(Mx+num_gp_x)+63)/64, 64>>>( array_device, array_device_deriv, time_step, Ny, Ly, num_gp_x, num_gp_y , size , Mx);
}

void cuda_array::deriv_real_x( uint time_step){
	deriv_real_x_cuda<<<((Ny+num_gp_y)*(Mx+num_gp_x)+63)/64, 64>>>( array_device, array_device_deriv, time_step, Ny, Ly, num_gp_x, num_gp_y , size, Mx, Lx );
}

void cuda_array::second_deriv_real_x( uint time_step){
	second_deriv_real_x_cuda<<<((Ny+num_gp_y)*(Mx+num_gp_x)+63)/64,64>>>( array_device, array_device_deriv, time_step, Ny, Ly, num_gp_x, num_gp_y , size, Mx, Lx );
}


void cuda_array::FFT_forward(uint time_step){
		
	if(cufftExecD2Z(plan_f, array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)
	, (cufftDoubleComplex*)array_device_freq + time_step*Mx*(Ny/2+1)) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecC2C creation failed\n");	
	}
	
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	

//	cufftDestroy(plan);	
}

void cuda_array::FFT_arakawa(uint time_step){
	if(cufftExecD2Z(plan_f, array_arakawa +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)
	, (cufftDoubleComplex*)array_arakawa_freq + time_step*Mx*(Ny/2+1)) != CUFFT_SUCCESS){
	fprintf(stderr, "CUFFT error: ExecC2C creation failed\n");	
	}
	
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	

}


void cuda_array::fill_array_test(uint time_step, double null){

	int i,iter;
	srand((unsigned)time(NULL));

	for(iter = 1; iter <= Mx; iter++){
		for(i=(iter*((Ny+num_gp_y)*num_gp_x*0.5) + num_gp_y/2); i < iter*((Ny+num_gp_y)*num_gp_x*0.5)+num_gp_y/2+Ny; i++){
			double new_i = i%(Ny+num_gp_y)-1.5; //for 2d gauss exactly in centre at -1.5
			double new_iter = iter %(Mx+num_gp_x)-0.5; //dito at -0.5 
			
			//array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = -cos(((double)((new_iter- (Mx+num_gp_x+num_gp_x/2+ (new_iter)*  (Mx+num_gp_x))) )/Mx)*Lx);  
			//array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = exp(-((new_i-Ny/2)*delta_y)*((new_i-Ny/2)*delta_y)/2);	
			//array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = exp(-((new_iter-Mx/2)*delta_x)*((new_iter-Mx/2)*delta_x)/10);	
			//array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = exp(-((new_iter-Mx/2)*delta_x)*((new_iter-Mx/2)*delta_x)/10) * ((-0.2*((new_iter-Mx/2)*delta_x))*(-0.2*((new_iter-Mx/2)*delta_x))-0.2);
			//double X = ((new_iter-Mx/2)*delta_x);
			//double Y = ((new_i - Ny/2)*delta_y);
			//printf(" %f\n",X);
			double X = ((new_iter)*delta_x);//-4*M_PI;
			double Y = ((new_i)*delta_y) ;//+ 0.25*Ly;
				
			if( null ==0){
			array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = 0;			
			//array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = sin(M_PI*Y)+1;//*sin(M_PI*Y)+1;
			//array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = (Lx-X)/Lx;			
			}
			
			else if( null == 7){
			array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = (Lx-X)/Lx;//exp(-3*(X)/(Lx));
			//-Y/Ly*exp(0.5*(-Y*Y));
			//array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = exp(-0.5*X*X - 0.5*Y*Y)  *(X*X+Y*Y-2);
			
			}
			
			else{
			array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = 1e-5*sin(2*M_PI*Y)*sin(M_PI*X)
			//if(iter<10){array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)]=0;}
			//exp(0.5*(-X*X-Y*Y));
			//array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = exp(-X*X - Y*Y)  *(X*X+Y*Y-2);
			//array_host[i+time_step*(Ny+num_gp_y)*(Mx+num_gp_x)] = //(Lx-X)/Lx + 
			//1e-2*cos(2*M_PI*Y/Ly)* sin(M_PI*X/Lx) 
			+ (double)rand()/(double)1e16;
	//printf("hgjkhkjhb is %d\n", new_iter);
			//array_host[i] = -2*sin(M_PI*X)*sin(M_PI*Y);
//		}
//		for(i=iter*((Ny+num_gp_y)*num_gp_x*0.5) + num_gp_y/2; i < iter*((Ny+num_gp_y)*num_gp_x*0.5)+num_gp_y/2+Ny; i++){
//			array_host[i] = cos(((double)(i- (22+ (iter-1)*  (Ny+num_gp_y)) )/Ny)*Ly); 

//	array_host[i] = 0;
	//printf("%d, %f\n", i, array_host[i]);
			}
		}
	}
/*	for(i = 0; i < size; i++){
		printf("%d, %f\n", i, array_host[i]);
	}
*/
}

void cuda_array::FFT_backward(uint time_step){
		if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) array_device_freq + time_step*(Ny/2+1)*Mx,  array_device +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + time_step*(Mx+num_gp_x)*(Ny+num_gp_y)) != CUFFT_SUCCESS){

	fprintf(stderr, "CUFFT error: ExecC2C creation failed");	
	}
	
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	normalize<<<((Ny+num_gp_y)*((Mx+num_gp_x))+63)/64, 64>>>( time_step, array_device, Mx, Ny, num_gp_x, num_gp_y);
}

void cuda_array::FFT_backward_deriv(uint time_step){
	if(cufftExecZ2D(plan_b,(cufftDoubleComplex*) array_device_freq_deriv+time_step*(Ny/2+1)*Mx,  array_device_deriv +  (Ny+num_gp_y)*num_gp_x/2 + num_gp_y/2 + (time_step %4)*(Mx+num_gp_x)*(Ny+num_gp_y)) != CUFFT_SUCCESS){

	fprintf(stderr, "CUFFT error: ExecC2C creation failed");	
	}
	
	if(cudaDeviceSynchronize() != cudaSuccess){
	fprintf(stderr, "CUFFT error: Synchronisation creation failed");
	}
	
	normalize<<<((Ny+num_gp_y)*((Mx+num_gp_x))+63)/32, 32>>>( time_step, array_device_deriv, Mx, Ny, num_gp_x, num_gp_y);
}

__global__
void clean_all_cuda( cufftDoubleReal *array_device, cufftDoubleReal *array_device_deriv,  uint size, uint time_step, uint Ny, uint num_gp_y, uint Mx, uint num_gp_x){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x;
	int first_value =  ((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)); 
	int last_value =  (Ny+num_gp_y)*(Mx+num_gp_x)+((time_step+1)*(Ny+num_gp_y)*(Mx+num_gp_x));

	if(index >= first_value && index < last_value){
		array_device[index] = 0;
		//array_device_deriv[index] = 0;
	}
}

__global__
void normalize( uint time_step, cufftDoubleReal *array_device, uint Mx, uint Ny, int num_gp_x, int num_gp_y){
	int index;
	double divisor = 1.0/(double)Ny;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	int first_value =  0;//((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)); 
	int last_value =  (Ny+num_gp_y)*(Mx+num_gp_x);


	if(index % (Ny+num_gp_y) == 0 ||  index % (Ny+num_gp_y) == 1 || index % (Ny+num_gp_y) == Ny+2 || index % (Ny+num_gp_y) == Ny+3 
			|| (index >= first_value && index <first_value + Ny+num_gp_y) || (index < last_value && index >= (last_value-Ny-num_gp_y))){
	}	
	else{
	//if (index < (Ny+num_gp_y)*(Mx+num_gp_x)){
		array_device[index + (time_step)*(Mx+num_gp_x)*(Ny+num_gp_y)] *= divisor;
	}
}

__global__
void deriv_real_y_cuda(cufftDoubleReal *array_device, cufftDoubleReal *array_device_deriv, uint time_step, uint Ny, double Ly, int num_gp_x, int num_gp_y , size_t size, uint Mx){
	double inv2dy = (Ny/(2*Ly));
	int index;
	
	int first_value = ( (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y)+(Ny+num_gp_y)*(Mx+num_gp_x)*time_step;
	int last_value = (Ny+num_gp_y)*(Mx+num_gp_x) * (time_step+1) -(  (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y);
//	index = blockIdx.x*blockDim.x + threadIdx.x;
	index = blockIdx.x*blockDim.x+threadIdx.x + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);

	//delete next line in final programm
//	array_device_deriv[index] = 0;

	if( index >= first_value && index < last_value){
		if((index) % (Ny+num_gp_y) == 0 || (index-1) % (Ny+num_gp_y) == 0 
		 || (index-Ny-2) % (Ny+num_gp_y) == 0 || (index-Ny-3) % (Ny+num_gp_y) == 0) {}
		else{			array_device_deriv[index] = inv2dy * (array_device[index+1] - array_device[index-1]);
//			array_device_deriv[index] = inv2dy * (array_device[index+1] - array_device[index-1]);
		}//printf("array_device_deriv[index] is %f\n", array_device_deriv[index]);
	}	
}

__global__
void deriv_real_x_cuda(cufftDoubleReal *array_device, cufftDoubleReal *array_device_deriv, uint time_step, uint Ny, double Ly, int num_gp_x, int num_gp_y , size_t size , uint Mx, double Lx){
	double inv2dx = (Mx/(2*Lx));
	int index;
	
	int first_value = ( (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y)+(Ny+num_gp_y)*(Mx+num_gp_x)*time_step;
	int last_value = (Ny+num_gp_y)*(Mx+num_gp_x) * (time_step+1) -(  (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y);
	index = blockIdx.x*blockDim.x+threadIdx.x + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);
		if( index >= first_value && index < last_value){
		if((index) % (Ny+num_gp_y) == 0 || (index-1) % (Ny+num_gp_y) == 0 
		 || (index-Ny-2) % (Ny+num_gp_y) == 0 || (index-Ny-3) % (Ny+num_gp_y) == 0) {}
		else{			array_device_deriv[index] = inv2dx * (array_device[index+Ny+num_gp_y] - array_device[index-Ny-num_gp_y]);
//			array_device_deriv[index] = inv2dy * (array_device[index+1] - array_device[index-1]);
		}//printf("array_device_deriv[index] is %f\n", array_device_deriv[index]);
	}	
}

__global__
void second_deriv_real_y_cuda(cufftDoubleReal *array_device, cufftDoubleReal *array_device_deriv, uint time_step, uint Ny, double Ly, int num_gp_x, int num_gp_y , size_t size, uint Mx){
	double invdy = (Ny/(Ly));
	int index;
	
	int first_value = ( (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y)+(Ny+num_gp_y)*(Mx+num_gp_x)*time_step;
	int last_value = (Ny+num_gp_y)*(Mx+num_gp_x) * (time_step+1) -(  (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y);
	index = blockIdx.x*blockDim.x+threadIdx.x + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);

	//delete next line in final programm
//	array_device_deriv[index] = 0;
//	array_device_deriv[index] = 0;
	if( index >= first_value && index < last_value){
		if((index) % (Ny+num_gp_y) == 0 || (index-1) % (Ny+num_gp_y) == 0 
		 || (index-Ny-2) % (Ny+num_gp_y) == 0 || (index-Ny-3) % (Ny+num_gp_y) == 0) {}
		else{			array_device_deriv[index] = invdy*invdy * (array_device[index+1] + array_device[index-1] - 2*array_device[index]);
//			array_device_deriv[index] = 0;
//			array_device_deriv[index] = inv2dy * (array_device[index+1] - array_device[index-1]);
		}//printf("array_device_deriv[index] is %f\n", array_device_deriv[index]);
	}	
}

__global__
void second_deriv_real_x_cuda(cufftDoubleReal *array_device, cufftDoubleReal *array_device_deriv, uint time_step, uint Ny, double Ly, int num_gp_x, int num_gp_y , size_t size , uint Mx, double Lx){
	double invdx = (Mx/(Lx));
	int index;
	
	int first_value = ( (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y)+(Ny+num_gp_y)*(Mx+num_gp_x)*time_step;
	int last_value = (Ny+num_gp_y)*(Mx+num_gp_x) * (time_step+1) -(  (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y);
//	index = blockIdx.x*blockDim.x + threadIdx.x;
	index = blockIdx.x*blockDim.x+threadIdx.x + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);

	//delete next line in final programm
//	array_device_deriv[index] = 0;

	if( index >= first_value && index < last_value){
		if((index) % (Ny+num_gp_y) == 0 || (index-1) % (Ny+num_gp_y) == 0 
		 || (index-Ny-2) % (Ny+num_gp_y) == 0 || (index-Ny-3) % (Ny+num_gp_y) == 0) {}
		else{			array_device_deriv[index] = invdx *invdx* (array_device[index+Ny+num_gp_y] + array_device[index-Ny-num_gp_y] - 2*array_device[index]);
//			array_device_deriv[index] = inv2dy * (array_device[index+1] - array_device[index-1]);
		}//printf("array_device_deriv[index] is %f\n", array_device_deriv[index]);
	}	
}

__global__
void deriv_freq_y_cuda(CuCmplx<double> *array_device_freq, CuCmplx<double> *array_device_freq_deriv , uint time_step, uint Ny, uint Mx, double Ly){
	int index;
//	index = blockIdx.x*blockDim.x + threadIdx.x;
	index = blockIdx.x*blockDim.x+threadIdx.x;// + time_step*(Ny)*(Mx);

	double spiDL = 2*M_PI/Ly;
//	array_device_freq_deriv[index] = 0;
	CuCmplx<double> i(0.0, 1.0);
	int first_value = 0;//(time_step)*Mx*(Ny/2+1);//-1;
	int last_value = Mx*(Ny/2+1);
//	array_device_freq[index] = 0;
//	printf("fgghjk %f\n", array_device_freq[index]);
	if ( index >= first_value && index < last_value ){//&& (index ) % Ny != 0 ){
		array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)] = array_device_freq[index+ time_step*Mx*(Ny/2+1)] *i* spiDL*(index %(Ny/2+1)) ;
//		array_device_freq_deriv[index] = 0;
	

	if ((index) % (Ny/2+1) == 0){
		array_device_freq_deriv[index] = 0;
	}
	
	}
}


__global__
void second_deriv_freq_y_cuda(CuCmplx<double> *array_device_freq, CuCmplx<double> *array_device_freq_deriv , uint time_step, uint Ny, uint Mx, double Ly){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x;// + time_step*(Ny)*(Mx);

	double spiDL = 2*M_PI/Ly;
	CuCmplx<double> i(0.0, 1.0);
	int first_value = (time_step)*Mx*(Ny/2+1);
	int last_value = (time_step+1)*Mx*(Ny/2+1);
//	printf("ghjkkjhghjk is %d\n", index);	
	if ( index > first_value && index < last_value && (index ) % Ny != 0 ){
		array_device_freq_deriv[index] = array_device_freq[index] * (-1)*spiDL*spiDL*(index %(Ny/2+1))*(index % (Ny/2+1)) ;
		printf("index: %d k: %f\n", index,(double) (index%(Ny/2+1)));
		//array_device_freq_deriv[index] = array_device_freq_deriv[index] * i*spiDL*(index %(Ny/2+1)) ;
		//array_device_freq_deriv[index] = array_device_freq[index] * i*i*spiDL*spiDL*(index % (Ny/2+1))*(index%(Ny/2+1));
	}
	if ((index) % Ny == 0){
		array_device_freq_deriv[index] = array_device_freq[index] * (-1)*spiDL*spiDL*(index %(Ny/2+1))*(index % (Ny/2+1)) ;
			printf("index: %d k: %f\n", index,(double) (index%(Ny/2+1)));
	
	//	array_device_freq_deriv[index] = array_device_freq[index] * i*spiDL*(index %(Ny/2+1)) ;
	//	array_device_freq_deriv[index] = array_device_freq_deriv[index] * i*spiDL*(index %(Ny/2+1)) ;
	}
}


__global__
void twod_QR_array_freq(CuCmplx<double>* array_device_freq ,cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, double delta_x, double bound_type_l, double bound_type_r, double BC_1, double BC_2, cuDoubleComplex r1, double delta_t, double diff_coeff){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x; 

//	double spiDL = 2*M_PI/Ly;
	int first_value = 0;
	int last_value  = Mx*(Ny/2+1);

	if ( index >= first_value && index < last_value  ){

	//	cuDoubleComplex *test = (cuDoubleComplex*)(&array_device_freq[index]);
	//	printf("test variable in kernel is %f\n", (*test).x);
		//	invert this 			to 	
	//	invert this 			to 	
	//	0	3	6	9		0	1	2	
	//	1	4	7	10		3	4	5		
	//	2	5	8	11		6	7	8
	//						9	10	11
		int row = index % (Ny/2+1);
		int column = (int)(index/(Ny/2+1));
		int index1 = row * (Mx) + column; 
		//double inv_dx_2 = 1./(delta_x*delta_x);

			
		d_b[index1].x = array_device_freq[index + time_step*Mx*(Ny/2+1)].re();
		d_b[index1].y = array_device_freq[index + time_step*Mx*(Ny/2+1)].im();	
		
		//printf("alagalg is %f\n", d_b[index1].x);

		if(bound_type_l == 0){
			if(column == 0 && row == 0){
			//if(column == 0){		
				//printf("index: %f\n", BC_1);//array_device_freq[index].re());
				//printf("############################################################BC: %f\n", r1.x);
				d_b[index1 ].x  = 
				d_b[index1].x +2*BC_1*r1.x*Ny;
				//array_device_freq[index + time_step*Mx*(Ny/2+1)].re() +2*BC_1*Ny;
		//printf("BC_1 : %f\n",BC_1);				
				//printf("index: %f\n", array_device_freq[index].re());
				
				//printf("index: %f\n", array_device_freq[index].re());
			}
		
		}
		else{
			if(column == 0 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x +delta_x*BC_1*r1.x*Ny;
			}
		
		}
		if(bound_type_r == 0){
			//printf("right: %f\n", bound_type_r);
			if(column == Mx-1 && row == 0){
			//if(column == Mx-1){
				//printf("index: %d\n", index);
				d_b[index1].x  = 
				d_b[index1].x +2*BC_2*r1.x*Ny;
				//array_device_freq[index + time_step*Mx*(Ny/2+1)].re() +2*BC_2*Ny;
			//printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hallo index: %f\n", array_device_freq[index].re());
			}
		}
		else{
			if(column == Mx-1 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x+ delta_x*BC_2*r1.x*Ny;
			//	printf("ihallo index: %d\n", index);
			}
		}

	/*	
		if(bound_type == 0){
			if(column == 0 && row == 0){
				printf("index: %f\n", d_b[index1].x);
				//printf("BC: %f\n", BC_1);
				d_b[index1].x  = d_b[index1].x -2*BC_1;
				printf("index: %f\n", d_b[index1].x);
			}
			if(column == Mx-1 && row == 0){
				d_b[index1].x = d_b[index1].x -2*BC_2;
			//	printf("index: %f\n", array_device_freq[index].re());
			}
		}
		else{
			if(column == 0 && row == 0){
				d_b[index1].x = d_b[index1].x +delta_x*BC_1;
			//	printf("index: %d\n", index);
			}
			if(column == Mx-1 && row == 0){
				d_b[index1].x = d_b[index1].x -delta_x*BC_2;
			//	printf("index: %d\n", index);
			}
		}
		*/
	
	}

}

__global__
void rhs_1(CuCmplx<double>* array_device_freq ,cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, CuCmplx<double>* array_arakawa_freq, CuCmplx<double>* array_device_freq_deriv, double delta_t, int order, double delta_x, double bound_type_l,double bound_type_r, double BC_1, double BC_2, cuDoubleComplex r){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x; 

//	double spiDL = 2*M_PI/Ly;
	int first_value = 0;//(time_step)*Mx*(Ny/2+1);
	int last_value = Mx*(Ny/2+1);
	//printf("ajfhlajbfh is %f\n", array_device_freq_deriv[index]);

	if ( index >= first_value && index < last_value  ){
	

		double K1;	double A1;
		double alpha0_inv;
		alpha0_inv = 1.0;
		K1 = -1.0;	A1 = 1.0;
	
		int row = index % (Ny/2+1);
		int column = (int)(index/(Ny/2+1));
		int index1 = row * (Mx) + column; 
		//double inv_dx_2 = 1./(delta_x*delta_x);
	
		
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t			
		+K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].re()*delta_t; 
	
		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].im()*delta_t; 
	
		if(bound_type_l == 0){
			if(column == 0 && row == 0){
				//printf("index: %f\n", BC_1);//array_device_freq[index].re());
				//printf("############################################################BC: %f\n", r1.x);
				d_b[index1].x  = 
				d_b[index1].x +2*BC_1*r.x*Ny*alpha0_inv;
				
				//printf("index: %f\n", array_device_freq[index].re());
				
				//printf("index: %f\n", array_device_freq[index].re());
			}
		}
		
		else{
			if(column == 0 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x +delta_x*BC_1*r.x*Ny*alpha0_inv;
			}
	
		}
		if(bound_type_r == 0){
			if(column == Mx-1 && row == 0){
				//printf("index: %d\n", index);
				d_b[index1].x  = 
				d_b[index1].x +2*BC_2*r.x*Ny*alpha0_inv;
			//printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hallo index: %f\n", array_device_freq[index].re());
			}
		}
		else{	
			if(column == Mx-1 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x+ delta_x*BC_2*r.x*Ny*alpha0_inv;
			//	printf("index: %d\n", index);
			}
		}
}
}


__global__
void rhs_2(CuCmplx<double>* array_device_freq ,cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, CuCmplx<double>* array_arakawa_freq, CuCmplx<double>* array_device_freq_deriv, double delta_t, int order, double delta_x, double bound_type_l,double bound_type_r, double BC_1, double BC_2, cuDoubleComplex r){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x; 

//	double spiDL = 2*M_PI/Ly;
	int first_value = 0;//(time_step)*Mx*(Ny/2+1);
	int last_value = Mx*(Ny/2+1);
	//printf("ajfhlajbfh is %f\n", array_device_freq_deriv[index]);

	if ( index >= first_value && index < last_value  ){
	

		double K1;	double A1;
		double K2;	double A2;
		double alpha0_inv;


		alpha0_inv = 2.0/3.0;
		K1 = -2.0;	A1 = 2.0;
		K2 = 1.0;	A2 = -0.5;
	
		int row = index % (Ny/2+1);
		int column = (int)(index/(Ny/2+1));
		int index1 = row * (Mx) + column; 
		//double inv_dx_2 = 1./(delta_x*delta_x);
		
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+A2*array_device_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()
	
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t
			
		+K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_device_freq_deriv[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t;


		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+A2*array_device_freq[index + ((time_step-1+4)%4)*Mx*(Ny/2+1)].im()
		
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t
	
		 +K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+ K2*array_device_freq_deriv[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t;
	

		if(bound_type_l == 0){
			if(column == 0 && row == 0){
				//printf("index: %f\n", BC_1);//array_device_freq[index].re());
				//printf("############################################################BC: %f\n", r1.x);
				d_b[index1].x  = 
				d_b[index1].x +2*BC_1*r.x*Ny*alpha0_inv;
				
				//printf("index: %f\n", array_device_freq[index].re());
				
				//printf("index: %f\n", array_device_freq[index].re());
			}
		}
		
		else{
			if(column == 0 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x +delta_x*BC_1*r.x*Ny*alpha0_inv;
			}
	
		}
		if(bound_type_r == 0){
			if(column == Mx-1 && row == 0){
				//printf("index: %d\n", index);
				d_b[index1].x  = 
				d_b[index1].x +2*BC_2*r.x*Ny*alpha0_inv;
			//printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hallo index: %f\n", array_device_freq[index].re());
			}
		}
		else{	
			if(column == Mx-1 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x+ delta_x*BC_2*r.x*Ny*alpha0_inv;
			//	printf("index: %d\n", index);
			}
		}

}
}


__global__
void rhs_without_deriv(CuCmplx<double>* array_device_freq ,cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, CuCmplx<double>* array_arakawa_freq, CuCmplx<double>* array_device_freq_deriv, double delta_t, int order, double delta_x, double bound_type_l,double bound_type_r, double BC_1, double BC_2, cuDoubleComplex r){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x; 

//	double spiDL = 2*M_PI/Ly;
	int first_value = 0;//(time_step)*Mx*(Ny/2+1);
	int last_value = Mx*(Ny/2+1);
	//printf("ajfhlajbfh is %f\n", array_device_freq_deriv[index]);

	if ( index >= first_value && index < last_value  ){
	
	
		int row = index % (Ny/2+1);
		int column = (int)(index/(Ny/2+1));
		int index1 = row * (Mx) + column; 
	
		double K1;	double A1;
		double K2;	double A2;
		double K3;	double A3;
		//double alpha0_inv;
		

		if(order == 1){
		//printf("1111111111111111111111111111111111111111111111111111111111111111111\n");
		//alpha0_inv = 1.0;
		K1 = -1.0;	A1 = 1.0;
						
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t;
		
		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t; 
			
		}
		
		else if(order == 2){
		//printf("222222222222222222222222222222222222222222222222222222222222222222222222\n");
		//alpha0_inv = 2.0/3.0;
		K1 = -2.0;	A1 = 2.0;
		K2 = 1.0;	A2 = -0.5;
			
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+A2*array_device_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()
	
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t;	

		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+A2*array_device_freq[index + ((time_step-1+4)%4)*Mx*(Ny/2+1)].im()
			
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t;
		}	
		
		else{
		//printf("3333333333333333333333333333333333333333333333333333333333333333333\n");	
		//alpha0_inv = 6.0/11.0;
		K1 = -3.0;	A1 = 3.0;
		K2 = 3.0;	A2 = -1.5;
		K3 = -1.0;	A3 = 1.0/3.0;
		
		
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+A2*array_device_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()
		+A3*array_device_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].re()
			
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t
		+K3*array_arakawa_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].re()*delta_t;
		

		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+A2*array_device_freq[index + ((time_step-1+4)%4)*Mx*(Ny/2+1)].im()
		+A3*array_device_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].im()			
			
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t
		+K3*array_arakawa_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].im()*delta_t;	
		}







		/*
		if(order == 1){
		//printf("1111111111111111111111111111111111111111111111111111111111111111111\n");
		alpha0_inv = 1.0;
		K1 = -1.0;	A1 = 1.0;
						
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t;

		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t; 
		
		}

		else if(order == 2){
		//printf("222222222222222222222222222222222222222222222222222222222222222222222222\n");
		alpha0_inv = 2.0/3.0;
		K1 = -2.0;	A1 = 2.0;
		K2 = 1.0;	A2 = -0.5;
			
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+A2*array_device_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()
	
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t;

		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+A2*array_device_freq[index + ((time_step-1+4)%4)*Mx*(Ny/2+1)].im()
			
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t;
		}

		else{
		//printf("3333333333333333333333333333333333333333333333333333333333333333333\n");	
		alpha0_inv = 6.0/11.0;
		K1 = -3.0;	A1 = 3.0;
		K2 = 3.0;	A2 = -1.5;
		K3 = -1.0;	A3 = 1.0/3.0;
		}
		
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+A2*array_device_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()
		+A3*array_device_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].re()
			
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t
		+K3*array_arakawa_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].re()*delta_t;			

		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+A2*array_device_freq[index + ((time_step-1+4)%4)*Mx*(Ny/2+1)].im()
		+A3*array_device_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].im()			
			
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t
		+K3*array_arakawa_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].im()*delta_t;	
		//}
		*/
		if(bound_type_l == 0){
			if(column == 0 && row == 0){
			//if(column == 0){
				d_b[index1].x  = 
				d_b[index1].x +2*BC_1*r.x*Ny;//*alpha0_inv;
				//printf("BC_1 %f\n", BC_1);	
			}
		}
		
		else{
			if(column == 0 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x +delta_x*BC_1*r.x*Ny;//*alpha0_inv;
			}
	
		}
		if(bound_type_r == 0){
			if(column == Mx-1 && row == 0){
			//if(column == Mx-1){
				//printf("index: %d\n", index);
				d_b[index1].x  = 
				d_b[index1].x +2*BC_2*r.x*Ny;//alpha0_inv;
			//printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hallo index: %f\n", array_device_freq[index].re());
			}
		}
		else{	
			if(column == Mx-1 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x+ delta_x*BC_2*r.x*Ny;//*alpha0_inv;
			//	printf("index: %d\n", index);
			}
		}
	}
}




__global__
void rhs(CuCmplx<double>* array_device_freq ,cuDoubleComplex* d_b, uint time_step, uint Ny, uint Mx, CuCmplx<double>* array_arakawa_freq, CuCmplx<double>* array_device_freq_deriv, double delta_t, int order, double delta_x, double bound_type_l,double bound_type_r, double BC_1, double BC_2, cuDoubleComplex r){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x; 

//	double spiDL = 2*M_PI/Ly;
	int first_value = 0;//(time_step)*Mx*(Ny/2+1);
	int last_value = Mx*(Ny/2+1);
	//printf("ajfhlajbfh is %f\n", array_device_freq_deriv[index]);

	if ( index >= first_value && index < last_value  ){
	
		int row = index % (Ny/2+1);
		int column = (int)(index/(Ny/2+1));
		int index1 = row * (Mx) + column; 
		//double inv_dx_2 = 1./(delta_x*delta_x);

		double K1;	double A1;
		double K2;	double A2;
		double K3;	double A3;
		//double alpha0_inv;

		if(order == 1){
		//printf("1111111111111111111111111111111111111111111111111111111111111111111\n");
		//alpha0_inv = 1.0;
		K1 = -1.0;	A1 = 1.0;
						
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].re()*delta_t;
		
		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].im()*delta_t;
			
		}
		
		else if(order == 2){
		//printf("222222222222222222222222222222222222222222222222222222222222222222222222\n");
		//alpha0_inv = 2.0/3.0;
		K1 = -2.0;	A1 = 2.0;
		K2 = 1.0;	A2 = -0.5;
			
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+A2*array_device_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()
	
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t
				
		+K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_device_freq_deriv[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t;
		

		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+A2*array_device_freq[index + ((time_step-1+4)%4)*Mx*(Ny/2+1)].im()
			
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t
		
		+K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K2*array_device_freq_deriv[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t;
		}	
		
		else{
		//printf("3333333333333333333333333333333333333333333333333333333333333333333\n");	
		////alpha0_inv = 6.0/11.0;
		K1 = -3.0;	A1 = 3.0;
		K2 = 3.0;	A2 = -1.5;
		K3 = -1.0;	A3 = 1.0/3.0;
		
		
		d_b[index1].x = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].re() 
		+A2*array_device_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()
		+A3*array_device_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].re()
			
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t
		+K3*array_arakawa_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].re()*delta_t
					
		+K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].re()*delta_t 
		+K2*array_device_freq_deriv[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].re()*delta_t
		+K3*array_device_freq_deriv[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].re()*delta_t;
		


		d_b[index1].y = A1*array_device_freq[index + time_step*Mx*(Ny/2+1)].im() 
		+A2*array_device_freq[index + ((time_step-1+4)%4)*Mx*(Ny/2+1)].im()
		+A3*array_device_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].im()			
			
		+K1*array_arakawa_freq[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K2*array_arakawa_freq[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t
		+K3*array_arakawa_freq[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].im()*delta_t
		
		+K1*array_device_freq_deriv[index + time_step*Mx*(Ny/2+1)].im()*delta_t 
		+K2*array_device_freq_deriv[index + ((time_step-1+4)%4 )*Mx*(Ny/2+1)].im()*delta_t
		+K3*array_device_freq_deriv[index + ((time_step-2+4)%4)*Mx*(Ny/2+1)].im()*delta_t;
		
		}

		if(bound_type_l == 0){
			if(column == 0 && row == 0){
			//if(column == 0){
				d_b[index1].x  = 
				d_b[index1].x +2*BC_1*r.x*Ny;//*alpha0_inv;
				//printf("BC_1 %f\n", BC_1);	
			}
		}
		
		else{
			if(column == 0 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x +delta_x*BC_1*r.x*Ny;//*alpha0_inv;
			}
	
		}
		if(bound_type_r == 0){
			if(column == Mx-1 && row == 0){
			//if(column == Mx-1){
				//printf("index: %d\n", index);
				d_b[index1].x  = 
				d_b[index1].x +2*BC_2*r.x*Ny;//alpha0_inv;
			//printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hallo index: %f\n", array_device_freq[index].re());
			}
		}
		else{	
			if(column == Mx-1 && row == 0){
				d_b[index1].x  = 
				d_b[index1].x+ delta_x*BC_2*r.x*Ny;//*alpha0_inv;
			//	printf("index: %d\n", index);
			}
		}
	}
}


__global__
void twod_QR_array_real(double* array_device, double* d_b, uint time_step, uint Ny, uint Mx , uint num_gp_x, uint num_gp_y, double dx){
	int index;
	int first_value = ( (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y);// + (Ny+num_gp_y)*(Mx+num_gp_x)*time_step;
	int last_value = (Ny+num_gp_y)*(Mx+num_gp_x)- (Ny+num_gp_y);//+(Ny+num_gp_y)*(Mx+num_gp_x)*(time_step+1);
//	index = blockIdx.x*blockDim.x + threadIdx.x;
	index = blockIdx.x*blockDim.x+threadIdx.x;

	if( index >= first_value && index < last_value){
	if((index) % (Ny+num_gp_y) == 0 || (index-1) % (Ny+num_gp_y) == 0 
	 || (index-Ny-2) % (Ny+num_gp_y) == 0 || (index-Ny-3) % (Ny+num_gp_y) == 0) {}
	else{	
		int row = (index - Ny*num_gp_x/2-num_gp_y-num_gp_y/2) % (Ny+num_gp_y);
		int column = (int)((index - Ny*num_gp_x/2-num_gp_y-num_gp_y/2)/((Ny+num_gp_y)));//+(Ny+num_gp_y)*(Mx+num_gp_x)*(time_step+1));
		int index1 = row * (Mx) + column; 
		
		d_b[index1] = array_device[index +  (Ny+num_gp_y)*(Mx+num_gp_x)*time_step]*dx*dx;
	
		if( (index1 % Mx) == 0){
			d_b[index1] = d_b[index1] - 1.0*dx;
	//		printf("index1: %d\n", index1); 
		}
		if ( (index1+1)%Mx == 0){
			d_b[index1] = d_b[index1] - 1.0*dx;
		}
		
		//printf("index: %d index1: %d\n", index, index1);
	}
}
}

__global__
void time_int_reorder(double* array_device, double* d_b, uint time_step, uint Ny, uint Mx , uint num_gp_x, uint num_gp_y){
	int index;
	int first_value = ( (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y);// + (Ny+num_gp_y)*(Mx+num_gp_x)*time_step;
	int last_value = (Ny+num_gp_y)*(Mx+num_gp_x)- (Ny+num_gp_y);//+(Ny+num_gp_y)*(Mx+num_gp_x)*(time_step+1);
//	index = blockIdx.x*blockDim.x + threadIdx.x;
	index = blockIdx.x*blockDim.x+threadIdx.x;

	if( index >= first_value && index < last_value){
	if((index) % (Ny+num_gp_y) == 0 || (index-1) % (Ny+num_gp_y) == 0 
	 || (index-Ny-2) % (Ny+num_gp_y) == 0 || (index-Ny-3) % (Ny+num_gp_y) == 0) {}
	else{	
		int row = (index - Ny*num_gp_x/2-num_gp_y-num_gp_y/2) % (Ny+num_gp_y);
		int column = (int)((index - Ny*num_gp_x/2-num_gp_y-num_gp_y/2)/((Ny+num_gp_y)));//+(Ny+num_gp_y)*(Mx+num_gp_x)*(time_step+1));
		int index1 = row * (Mx) + column; 
		
		d_b[index1] = array_device[index +  (Ny+num_gp_y)*(Mx+num_gp_x)*time_step];

	}
	}
}
__global__
void re_arrange_array_real(double* d_x, double* d_x1, uint time_step, uint Ny, uint Mx , uint num_gp_x, uint num_gp_y){

	int index;
	int first_value = ( (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y);// + (Ny+num_gp_y)*(Mx+num_gp_x)*time_step;
	int last_value = (Ny+num_gp_y)*(Mx+num_gp_x)- (Ny+num_gp_y);//+(Ny+num_gp_y)*(Mx+num_gp_x)*(time_step+1);
//	index = blockIdx.x*blockDim.x + threadIdx.x;
	index = blockIdx.x*blockDim.x+threadIdx.x;

	if( index >= first_value && index < last_value){
	if((index) % (Ny+num_gp_y) == 0 || (index-1) % (Ny+num_gp_y) == 0 
	 || (index-Ny-2) % (Ny+num_gp_y) == 0 || (index-Ny-3) % (Ny+num_gp_y) == 0) {}
	else{	
		int row = (index - Ny*num_gp_x/2-num_gp_y-num_gp_y/2) % (Ny+num_gp_y);
		int column = (int)((index - Ny*num_gp_x/2-num_gp_y-num_gp_y/2)/((Ny+num_gp_y)));//+(Ny+num_gp_y)*(Mx+num_gp_x)*(time_step+1));
		int index1 = row * (Mx) + column; 	
		d_x1[index] = d_x[index1];
		//printf("index: %d index1: %d\n", first_value, last_value);
	}
}
}

__global__
void re_arrange_array_freq(cuDoubleComplex* d_x, cuDoubleComplex* d_x1, uint time_step, uint Ny, uint Mx){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x; 
	int first_value = 0*Mx*(Ny/2+1);
	int last_value = Mx*(Ny/2+1);

	if ( index >= first_value && index < last_value  ){
		int row = index % (Ny/2+1);
		int column = (int)(index/(Ny/2+1));
		int index1 = row * (Mx) + column; 
		
		d_x1[index] = d_x[index1];
		//printf("d_x1 = %f\n", d_x1[index].x);
	}
}

__global__
void dirichlet_cuda(uint time_step, double *array_device, uint Mx, uint Ny, int num_gp_x, int num_gp_y, size_t size, double value1, double value2){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);
//	printf("ghjghj %d\n", index);//(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1);
	int first_value =  ((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1); 
	int last_value =  ((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1)+ Ny + num_gp_y+1;

	if(index > first_value &&  index < last_value){
//	printf("ghKJHGKGVGVHGKGHHGjghj %d\n", index);
	array_device[index] = 2*value1 - array_device[index+Ny+num_gp_y];
	}

	first_value =  ((time_step+1)*(Ny+num_gp_y)*(Mx+num_gp_x)-1 - Ny -num_gp_y); 
	last_value =  ((time_step+1)*(Ny+num_gp_y)*(Mx+num_gp_x));
	
	if ( index > first_value && index < last_value){
	array_device[index] = 2*value2 - array_device[index-Ny-num_gp_y];
	}	
}

__global__
void neumann_cuda(uint time_step, double *array_device_deriv, uint Mx, uint Ny, int num_gp_x, int num_gp_y, size_t size, double value1, double value2, double Lx){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);
//	printf("ghjghj %d\n", index);//(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1);
	int first_value =  ((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1); 
	int last_value =  ((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1)+ Ny + num_gp_y+1;
	double delta_x = (double)Mx/Lx;

	if(index > first_value &&  index < last_value){
//	printf("ghKJHGKGVGVHGKGHHGjghj %d\n", index);
	array_device_deriv[index] = array_device_deriv[index+Ny+num_gp_y] - delta_x*value1;
	}

	first_value =  ((time_step+1)*(Ny+num_gp_y)*(Mx+num_gp_x)-1 - Ny -num_gp_y); 
	last_value =  ((time_step+1)*(Ny+num_gp_y)*(Mx+num_gp_x));
	
	if ( index > first_value && index < last_value){
	array_device_deriv[index] = array_device_deriv[index-Ny-num_gp_y] + delta_x*value2;
	}	
}

__global__
void BC_array_device(double* array_device, uint time_step, uint Mx, uint Ny, uint num_gp_x, uint num_gp_y, double BC1, double BC2, double Lx){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);
//	printf("ghjghj %d\n", index);//(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1);
	int first_value =  ((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)); 
	int last_value =  ((time_step+1)*(Ny+num_gp_y)*(Mx+num_gp_x));

	if(index >= first_value &&  index < last_value){
		if((int)(index - time_step*(Ny+num_gp_y)*(Mx+num_gp_x)) / (num_gp_y + Ny) == 1){
			double inv_dx_2 = (Mx/Lx)*(Mx/Lx);
			array_device[index] = array_device[index]- inv_dx_2*2 *BC1;
			//printf("index: %d\n", index);
		}
		if((int)(index - time_step*(Ny+num_gp_y)*(Mx+num_gp_x)) / (num_gp_y + Ny) == Mx){
			double inv_dx_2 = (Mx/Lx)*(Mx/Lx);
			array_device[index] = array_device[index] - inv_dx_2*2*BC2;
			//printf("index: %d\n", index);
		}
	}
}


__global__
void prepare_gp_kernel(double* array_device, uint time_step, uint Mx, uint Ny, uint num_gp_x, uint num_gp_y,double bound_type_l, double bound_type_r, double BC1, double BC2, double Lx, double delta_x){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x;// + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);
//	printf("ghjghj %d\n", index);//(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1);
	int first_value =  0;//((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)); 
	int last_value =  (Ny+num_gp_y)*(Mx+num_gp_x);

	if(index >= first_value &&  index < last_value){
		//printf("lajghlkfaf %d\n", index);
	
		
		if(index % (Ny+num_gp_y) == 0 ||  index % (Ny+num_gp_y) == 1 || index % (Ny+num_gp_y) == Ny+2 || index % (Ny+num_gp_y) == Ny+3 
				|| (index >= first_value && index <first_value + Ny+num_gp_y) || (index < last_value && index >= (last_value-Ny-num_gp_y))){
			
			array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = 0;
		
			//printf("index: %d\n", index);
		}
		
		
		
		if(index % (Ny+num_gp_y) == 0 ){
			array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = 
			array_device[index + Ny + (time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)];
	
		}
		if(index % (Ny+num_gp_y) == 1 ){
			array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = 
			array_device[index + Ny + (time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)];
		}
		if(index % (Ny+num_gp_y) == Ny+2 ){
			array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = 
			array_device[index - Ny + (time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)];
		}
		if(index % (Ny+num_gp_y) == Ny+3 ){
			array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = 
			array_device[index - Ny + (time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)];	
		}
		
		///derichlett
			
		if(bound_type_l ==0){
			if(index >= first_value && index <first_value + Ny+num_gp_y  ){
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = 2*BC1 - 
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)+Ny+num_gp_y];  
			
				
				//array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = 0;//2*BC1 - 
				//printf("bc1: %f\n", BC1);
				//array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)+Ny+num_gp_y];  
					}
		}
		//neumann
		else{
		//	if(index < last_value && index >= (last_value-Ny-num_gp_y)){
			if(index >= first_value && index <first_value + Ny+num_gp_y){
		//		array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = delta_x*BC1 + 
		//		array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-Ny-num_gp_y];  
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = -delta_x*BC1 + 
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)+Ny+num_gp_y];  	
			}	
		}	
		
		if(bound_type_r ==0){
			if(index < last_value && index >= (last_value-Ny-num_gp_y)){
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = 2*BC2 - 
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-Ny-num_gp_y];  
			
		
			}
		}
		//neumann
		else{
			if(index < last_value && index >= (last_value-Ny-num_gp_y)){
				//array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = -delta_x*BC2 + 
				//array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-Ny-num_gp_y];  
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = +delta_x*BC2 + 
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-Ny-num_gp_y];		
			}	
		}
	}
}


__global__
void arakawa_kernel(double* a, double* b, double* array_arakawa, double dx, double dy, uint time_step, uint Mx, uint Ny, uint num_gp_x, uint num_gp_y){
	
	/*
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);
	int first_value =  ((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)) + ( (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y); 
	int last_value =  ((time_step+1)*(Ny+num_gp_y)*(Mx+num_gp_x)) -  ( (Ny + num_gp_y) * 0.5*num_gp_x + 0.5* num_gp_y);

	if(index >= first_value &&  index < last_value){
	if((index) % (Ny+num_gp_y) == 0 || (index-1) % (Ny+num_gp_y) == 0 
	 || (index-Ny-2) % (Ny+num_gp_y) == 0 || (index-Ny-3) % (Ny+num_gp_y) == 0) {}
	else{
	//printf("index: %d\n", index);
	
	double inv2dx = 1./(2.*dx);

	array_arakawa[index] = inv2dx*(a[index+Ny+num_gp_y]-a[index-Ny-num_gp_y])* inv2dx*(b[index+1]-b[index-1]) -
				 inv2dx*(b[index+Ny+num_gp_y]-b[index-Ny-num_gp_y])* inv2dx*(a[index+1]-a[index-1]);
	}	
	*/
	
		
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);
//	printf("ghjghj %d\n", index);//(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1);
	int first_value =  ((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)); 
	int last_value =  (time_step+1)*(Ny+num_gp_y)*(Mx+num_gp_x);

	if(index >= first_value &&  index < last_value){
		//printf("lajghlkfaf %d\n", index);
	
		
		if(index % (Ny+num_gp_y) == 0 ||  index % (Ny+num_gp_y) == 1 || index % (Ny+num_gp_y) == Ny+2 || index % (Ny+num_gp_y) == Ny+3 
				|| (index >= first_value && index <first_value + Ny+num_gp_y) || (index < last_value && index >= (last_value-Ny-num_gp_y))){
			
			//array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = 0;
		
			//printf("index: %d\n", index);
		}
	else{	
		
	

	array_arakawa[index] = (-1)/(12*dx*dy) * ((b[index-1] + b[index+Ny+num_gp_y-1] - b[index+1] - b[index+Ny+num_gp_y+1])*
				(a[index+Ny+num_gp_y] + a[index]) - (b[index-Ny-num_gp_y-1] + b[index-1] - b[index-Ny-num_gp_y+1] - b[index+1])*
				(a[index] + a[index-Ny-num_gp_y]) + (b[index+Ny+num_gp_y] + b[index + Ny+num_gp_y+1]- b[index-num_gp_y-Ny]-
				b[index-Ny-num_gp_y+1])*
				(a[index+1]+a[index]) - (b[index+Ny+num_gp_y-1] + b[index+Ny+num_gp_y] - b[index-Ny-num_gp_y-1]-b[index-Ny-num_gp_y])*
				(a[index] + a[index-1]) + (b[index+Ny+num_gp_y]-b[index+1])*(a[index+Ny+num_gp_y+1]+a[index])
				- (b[index-1]-b[index-Ny-num_gp_y])*(a[index] + a[index-Ny-num_gp_y-1]) + (b[index+1] - b[index-Ny-num_gp_y])*
				(a[index-Ny-num_gp_y+1]+a[index]) - (b[index+Ny+num_gp_y]-b[index-1])*(a[index] + a[index+Ny+num_gp_y-1]));

		}
	
	}
}



__global__
void gp_device(double* array_device, uint time_step, uint Mx, uint Ny, uint num_gp_x, uint num_gp_y,double bound_type_l, double bound_type_r, double BC1, double BC2, double Lx, double delta_x, cuDoubleComplex r  ){
	int index;
	index = blockIdx.x*blockDim.x+threadIdx.x;// + time_step*(Ny+num_gp_y)*(Mx+num_gp_x);
//	printf("ghjghj %d\n", index);//(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-1);
	int first_value =  0;//((time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)); 
	int last_value =  (Ny+num_gp_y)*(Mx+num_gp_x);

	if(index >= first_value &&  index < last_value){	
		///derichlett
		//printf("rq.x = %f\n", bound_type_l);	
			
		if(bound_type_l ==0){
			if(index >= (first_value + Ny+num_gp_y)&& index <first_value + (Ny+num_gp_y)*2){
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)+Ny+num_gp_y] +2.0*r.x*BC1 ;  
		//printf("rq.x = %f\n", 2.0*r.x*BC1);	
					}
		}
		//neumann
		else{
			if(index >= (first_value + Ny+num_gp_y)&& index <first_value + (Ny+num_gp_y)*2){
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-Ny-num_gp_y] +delta_x*BC1*r.x;  
		//printf("rq.x = %f\n", delta_x*BC1*r.x);	
			
			}	
		}	
		
		
		
		if(bound_type_r ==0){
			if(index < last_value -Ny-num_gp_y&& index >= (last_value-(Ny+num_gp_y)*2)){
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-Ny-num_gp_y] +2.0*r.x*BC2;  
			//printf("rq.x = %f\n", 2.0*r.x*BC1);	
			
		
			}
		}
		//neumann
		else{
			if(index < last_value -Ny-num_gp_y&& index >= (last_value-(Ny+num_gp_y)*2)){
				array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)] = array_device[index+(time_step)*(Ny+num_gp_y)*(Mx+num_gp_x)-Ny-num_gp_y] +delta_x*BC2*r.x;  
		//printf("sflgalgkalglgrq.x = %f\n", delta_x*r.x*BC2);	
		
			}	
		}
	}
}

