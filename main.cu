#include "datatype.h"
#include <sstream>
#include <H5Cpp.h>
#include "output.h"
#include "diagnostics.h"
//#include "slab.h"

using namespace H5;


int main(){
	uint timesteps =5000000;
	uint Mx = 128; 
	uint Ny = 128;
	double Lx = 1;
	double Ly = 1;//2*sqrt(2);
	int num_gp_x = 2;
	int num_gp_y = 4;
	double delta_t = 5e-3;
	double diff_coeff = 0.00158;
	double boundary_type_l_n= 1; //0 = derichlet, else= neumann
	double boundary_type_r_n= 0; //0 = derichlet, else= neumann
	double BC1_n = 1.0;
	double BC2_n = 0.0;
	double boundary_type_l= 0; //0 = derichlet, else= neumann
	double boundary_type_r= 0; //0 = derichlet, else= neumann
	double BC1 = 0.0;
	double BC2 = 0.0;

	cuda_array n(Mx,Ny ,Lx,Ly, num_gp_x, num_gp_y,delta_t, diff_coeff, boundary_type_l_n,boundary_type_r_n,BC1_n,BC2_n); // ...delta_t, diff_coef, boud., BC1, BC2
	cuda_array omega(Mx,Ny ,Lx,Ly, num_gp_x, num_gp_y,delta_t, diff_coeff, boundary_type_l,boundary_type_r,BC1,BC2); // ...delta_t, diff_coef, boud., BC1, BC2
	cuda_array phi(Mx,Ny ,Lx,Ly, num_gp_x, num_gp_y,delta_t, diff_coeff, boundary_type_l,boundary_type_r,BC1,BC2); // ...delta_t, diff_coef, boud., BC1, BC2
	//test.arakawa(0, test.array_device, example.array_device);	



	//output stuff///////////////////////////////////////////////////////////////////////////
		double* profile_n;
		profile_n = (double*)malloc(sizeof(double)*n.Mx);
		
		//double* fluc_n;
		//fluc_n = (double*)malloc(sizeof(double)*(n.Mx+n.num_gp_x)*(n.Ny+n.num_gp_y));
		
		double* profile_phi;
		profile_phi = (double*)malloc(sizeof(double)*n.Mx);
		
		//double* deriv_phi;
		//deriv_phi = (double*)malloc(sizeof(double)*(n.Mx+n.num_gp_x)*(n.Ny+n.num_gp_y));
	
		//double* fluc_phi;
		//fluc_phi = (double*)malloc(sizeof(double)*(n.Mx+n.num_gp_x)*(n.Ny+n.num_gp_y));
		
		ofstream profile_n_txt;
		ofstream profile_phi_txt;
		//ofstream deriv_phi_txt;
		//ofstream fluc_n_txt;	
		//ofstream fluc_phi_txt;		
		ofstream points_n;
		ofstream points_omega;
		ofstream points_phi;
	/////////////////////////////////////////////////////////////////////////////////////////

	output_h5 myout(Ny + n.num_gp_y, Mx+n.num_gp_x);	
	n.fill_array_test(0, 7);	//time_step
	omega.fill_array_test(0,0);
	phi.fill_array_test(0,1);

	n.copy_host_to_device(0);
	omega.copy_host_to_device(0);
	phi.copy_host_to_device(0);
	/*	
	n.prepare_gp(0);
	omega.prepare_gp(0);
	n.arakawa( 0, n.array_device, omega.array_device);
	n.copy_arakawa_to_host(0);
	*/	
	
	n.FFT_forward(0);
	omega.FFT_forward(0);
	phi.FFT_forward(0);
	
	//n.prepare_gp(0);
		
	/*
	n.prepare_gp(0);
	//omega.deriv_real_y(0, n.array_device);
	omega.deriv_freq_y( 0,n.array_device_freq );
	*/
	//int iter = 0;	
	
	//phi.QR_factorisation_2d(0,n.array_device_freq);
		
	//n.QR_factorisation_2d(0, n.array_device_freq);
	//n.copy_device_to_host(0);	
		
	//int i = 0;
	//int iter =0;
				
		//myout.surface(twodads::output_t::o_theta,omega.return_pointer_host(), 0.0);
		//myout.increment_output_counter();
			
			myout.surface(twodads::output_t::o_theta,n.return_pointer_host(), 0.0);
			myout.surface(twodads::output_t::o_omega,omega.return_pointer_host(), 0.0);
			myout.surface(twodads::output_t::o_strmf,phi.return_pointer_host(), 0.0);
			myout.increment_output_counter();
					
			profile_n_txt.open("profile_n.txt");
			for(int i=0; i< n.Mx; i++){
				profile_n_txt << profile_n[i]<<"\t " ;
				}
			profile_n_txt <<"\n " ;
			profile_n_txt.close();
			/*
			fluc_n_txt.open("fluc_n.txt");
			for(int i=0; i< n.Ny; i++){
				for(int iter = 0; iter < n.Mx; iter++){
				fluc_n_txt << fluc_n[n.adress(0//20 % 4
				, iter,i)]<<"\t " ;
				}
				fluc_n_txt <<"\n " ;
			}
			fluc_n_txt.close();
			*/
			profile_phi_txt.open("profile_phi.txt");
			for(int i=0; i< n.Mx; i++){
				profile_phi_txt << profile_phi[i]<<"\t " ;
				}
			profile_phi_txt <<"\n " ;
			profile_phi_txt.close();
			/*
			deriv_phi_txt.open("deriv_phi.txt");	
			for(int i=0; i< n.Ny; i++){
				for(int iter = 0; iter < n.Mx; iter++){
				deriv_phi_txt << deriv_phi[n.adress(0//20 % 4
				, iter,i)]<<"\t " ;
				}
				deriv_phi_txt <<"\n " ;
			}
			deriv_phi_txt.close();

			fluc_phi_txt.open("fluc_phi.txt");
			for(int i=0; i< n.Ny; i++){
				for(int iter = 0; iter < n.Mx; iter++){
				fluc_phi_txt << fluc_phi[n.adress(0//20 % 4
				, iter,i)]<<"\t " ;
				}
				fluc_phi_txt <<"\n " ;
			}
			fluc_phi_txt.close();
			*/
			points_phi.open("points_phi.txt");
			for(int iter = 0; iter < 3; iter++){
				points_phi << phi.array_host[n.adress(0//20 % 4
				//, iter*n.Mx/8,n.Ny/2)]<<"\t " ;
				, n.Mx/2,n.Ny/2+(iter-1))]<<"\t " ;
			}

				points_phi <<"\n " ;
			points_phi.close();
			/*	
			points_omega.open("points_omega.txt");
			for(int iter = 0; iter < 9; iter++){
				points_omega << omega.array_host[n.adress(0//20 % 4
				, iter*n.Mx/8,n.Ny/2)]<<"\t " ;
			}
				points_omega <<"\n " ;
			points_omega.close();
			*/
			points_n.open("points_n.txt");
			for(int iter = 0; iter < 9; iter++){
				points_n << n.array_host[n.adress(0//20 % 4
				, iter*n.Mx/8,n.Ny/2)]<<"\t " ;
			}
				points_n <<"\n " ;
			points_n.close();
			

	
	for(int i= 0; i<2; i++){
		n.FFT_backward(i);
		omega.FFT_backward(i);
		phi.FFT_backward(i);

		//phi.copy_host_to_device(i);	

		//dn/dt
		n.prepare_gp(i);
		phi.prepare_gp(i);
		n.arakawa( i, phi.array_device, n.array_device);
		n.FFT_arakawa( i);
		//n.deriv_freq_y(i, phi.array_device_freq);
		//n.copy_arakawa_to_host(i);

		if(i==0){
		//printf("bc_1 in main is %f\n", n.BC_1);
			//n.diffusion_2d(i);
			n.ssti_start(i, 1, 0);// 0 = without deriv, 1 = with deriv
		}
		if(i==1){
			//n.diffusion_2d(i);
			n.ssti_start(i, 2, 0);
		}
		
	
		//domega/dt
		//omega.copy_host_to_device(i);
		omega.prepare_gp(i);
		n.prepare_gp(i);
		omega.arakawa( i, phi.array_device, omega.array_device);
		omega.FFT_arakawa( i);
		//omega.copy_arakawa_to_host(i);	
		omega.deriv_freq_y( i,n.array_device_freq );
		//omega.deriv_real_y( i, n.array_device);
		//omega.copy_device_to_host(i);
		if(i==0){
		//printf("bc_1 in main is %f\n", n.BC_1);
			//omega.diffusion_2d(i);
			omega.ssti_start(i, 1, 1);// 0 = without deriv, 1 = with deriv
		}
		if(i==1){
			//omega.diffusion_2d(i);
			omega.ssti_start(i, 2, 1);
		}
	


		//laplace
		//phi.copy_host_to_device(i);	
		phi.QR_factorisation_2d(i, omega.array_device_freq);
		gpuErrchk(cudaMemcpy(phi.array_device_freq + (phi.Mx)*(phi.Ny/2+1)*((i+1)%4),
		phi.array_device_freq + (phi.Mx)*(phi.Ny/2+1)*(i),
		sizeof(CuCmplx<double>)*(phi.Mx)*(phi.Ny/2+1), cudaMemcpyDeviceToDevice));

		//printf("iter = %d\n", i);
		printf("time_step %d of %d\n", i, timesteps);
		
		
		n.FFT_backward(i);
		omega.FFT_backward(i);
		phi.FFT_backward(i);
		
		n.copy_device_to_host(i);	
		phi.copy_device_to_host(i);	
		omega.copy_device_to_host(i);		

		
		}

	
	for(int i= 2; i<timesteps+1; i++){
		int iter = i % 4;
			
		n.FFT_backward(iter);
		omega.FFT_backward(iter);
		phi.FFT_backward(iter);

		//phi.copy_host_to_device(iter);	
	
		//dn/dt
		n.prepare_gp(iter);
		phi.prepare_gp(iter);
		n.arakawa( iter, phi.array_device, n.array_device);
		//n.copy_arakawa_to_host(iter);
		n.FFT_arakawa( iter);
		//n.deriv_freq_y(iter, phi.array_device_freq);
		//n.clean_all( (iter ));
		n.ssti_without_deriv(iter);
		//n.ssti(iter);
		//n.diffusion_2d(iter);
		//n.ssti_start( iter, 2, 0);

		//domega/dt
		//omega.copy_host_to_device(iter);			
		omega.prepare_gp(iter);
		//n.prepare_gp(iter, 0,0);
		omega.arakawa( iter, phi.array_device, omega.array_device);
		//omega.copy_arakawa_to_host(iter);
		omega.FFT_arakawa( iter);
		omega.deriv_freq_y( iter, n.array_device_freq );
		//omega.deriv_real_y( iter, n.array_device);
		//omega.copy_device_to_host(iter);
		omega.ssti(iter);
		//omega.ssti_start(iter, 1,0);
		//omega.diffusion_2d(iter);

		//n.copy_device_to_host(iter);	
			
		//laplace
		//phi.copy_host_to_device(iter);		
		phi.QR_factorisation_2d(iter, omega.array_device_freq);
		gpuErrchk(cudaMemcpy(phi.array_device_freq + (phi.Mx)*(phi.Ny/2+1)*((iter+1)%4),
		phi.array_device_freq + (phi.Mx)*(phi.Ny/2+1)*(iter),
		sizeof(CuCmplx<double>)*(phi.Mx)*(phi.Ny/2+1), cudaMemcpyDeviceToDevice));

	
		//myout.surface(twodads::output_t::o_theta,n.return_pointer_host(), 0.0);
		//myout.increment_output_counter();
	
		if(i %100 ==0){
			printf("time_step %d of %d\n", i, timesteps);
					
			n.FFT_backward(iter);
			omega.FFT_backward(iter);
			phi.FFT_backward(iter);
				
			n.copy_device_to_host(iter);	
			phi.copy_device_to_host(iter);	
			omega.copy_device_to_host(iter);		
					
			myout.surface(twodads::output_t::o_theta,n.return_pointer_host(), 0.0);
			myout.surface(twodads::output_t::o_omega,omega.return_pointer_host(), 0.0);
			myout.surface(twodads::output_t::o_strmf,phi.return_pointer_host(), 0.0);
			myout.increment_output_counter();
		
			profile(n.array_host, profile_n, n.Mx, n.num_gp_x, n.Ny, n.num_gp_y);
			//fluc( n.array_host,profile_n, fluc_n, n.Mx, n.num_gp_x, n.Ny, n.num_gp_y);
			
			profile(phi.array_host, profile_phi, n.Mx, n.num_gp_x, n.Ny, n.num_gp_y);
			//fluc( phi.array_host,profile_phi, fluc_phi, n.Mx, n.num_gp_x, n.Ny, n.num_gp_y);
			//deriv_y(fluc_phi,deriv_phi, n.Mx, n.num_gp_x, n.Ny, n.num_gp_y, n.delta_y);
			
			////////////////////////////////////////////////////////////////////////
	
			profile_n_txt.open("profile_n.txt", ofstream::app);
			for(int i=0; i< n.Mx; i++){
				profile_n_txt << profile_n[i]<<"\t " ;
				}
			profile_n_txt <<"\n " ;
			profile_n_txt.close();
			/*
			fluc_n_txt.open("fluc_n.txt",ofstream::app);
			for(int i=0; i< n.Ny; i++){
				for(int iter = 0; iter < n.Mx; iter++){
				fluc_n_txt << fluc_n[n.adress(0//20 % 4
				, iter,i)]<<"\t " ;
				}
				fluc_n_txt <<"\n " ;
			}
			fluc_n_txt.close();
			*/
			profile_phi_txt.open("profile_phi.txt", ofstream::app);
			for(int i=0; i< n.Mx; i++){
				profile_phi_txt << profile_phi[i]<<"\t " ;
				}
			profile_phi_txt <<"\n " ;
			profile_phi_txt.close();
			/*
			deriv_phi_txt.open("deriv_phi.txt", ofstream::app);	
			for(int i=0; i< n.Ny; i++){
				for(int iter = 0; iter < n.Mx; iter++){
				deriv_phi_txt << deriv_phi[n.adress(0//20 % 4
				, iter,i)]<<"\t " ;
				}
				deriv_phi_txt <<"\n " ;
			}
			deriv_phi_txt.close();

			fluc_phi_txt.open("fluc_phi.txt",ofstream::app);
			for(int i=0; i< n.Ny; i++){
				for(int iter = 0; iter < n.Mx; iter++){
				fluc_phi_txt << fluc_phi[n.adress(0//20 % 4
				, iter,i)]<<"\t " ;
				}
				fluc_phi_txt <<"\n " ;
			}
			fluc_phi_txt.close();
			
			points_phi.open("points_phi.txt", ofstream::app);
			for(int iter = 0; iter < 9; iter++){
				points_phi << phi.array_host[n.adress(0//20 % 4
				//, iter*n.Mx/8,n.Ny/2)]<<"\t " ;
				, n.Mx/2,n.Ny/2+(iter-1))]<<"\t " ;
			}
				points_phi <<"\n " ;
			points_phi.close();
			
			points_omega.open("points_omega.txt", ofstream::app);
			for(int iter = 0; iter < 9; iter++){
				points_omega << omega.array_host[n.adress(0//20 % 4
				, iter*n.Mx/8,n.Ny/2)]<<"\t " ;
			}
				points_omega <<"\n " ;
			points_omega.close();

			points_n.open("points_n.txt", ofstream::app);
			for(int iter = 0; iter < 9; iter++){
				points_n << n.array_host[n.adress(0//20 % 4
				, iter*n.Mx/8,n.Ny/2)]<<"\t " ;
			}
				points_n <<"\n " ;
			points_n.close();
			*/


		}
		
		if(i %10 ==0){
			points_phi.open("points_phi.txt", ofstream::app);
			for(int iter = 0; iter < 3; iter++){
				points_phi << phi.array_host[n.adress(0//20 % 4
				//, iter*n.Mx/8,n.Ny/2)]<<"\t " ;
				, n.Mx/2,n.Ny/2+(iter-1))]<<"\t " ;
			}
				points_phi <<"\n " ;
			points_phi.close();
	
			points_n.open("points_n.txt", ofstream::app);
			for(int iter = 0; iter < 9; iter++){
				points_n << n.array_host[n.adress(0//20 % 4
				, iter*n.Mx/8,n.Ny/2)]<<"\t " ;
			}
				points_n <<"\n " ;
			points_n.close();
		
		}
	}	


		free(profile_n);free(profile_phi);
		//free(deriv_phi);
		//free(fluc_n);free(fluc_phi);

		ofstream outputFile1;
		outputFile1.open("n.txt");
		for(int i=0; i< n.Ny; i++){
			for(int iter = 0; iter < n.Mx; iter++){
			outputFile1 << n.array_host[n.adress(0//20 % 4
			, iter,i)]<<"\t " ;
			}
		outputFile1 <<"\n " ;
		}

		ofstream outputFile2("omega.txt");
		for(int i=0; i< n.Ny; i++){
			for(int iter = 0; iter < omega.Mx; iter++){
			outputFile2 << omega.array_host[n.adress(0//20 % 4
			, iter,i)]<<"\t " ;
			}
		outputFile2 <<"\n " ;
		}
		ofstream outputFile3("phi.txt");
		for(int i=0; i< n.Ny; i++){
			for(int iter = 0; iter < n.Mx; iter++){
			outputFile3 << phi.array_host[n.adress(0//20 % 4
			, iter,i)]<<"\t " ;
			}
		outputFile3 <<"\n " ;
		}

	//printf("hallo\n");
	//myout.surface(twodads::output_t::o_theta,n.return_pointer_host(), 0.0);
	//myout.increment_output_counter();
//	myout.surface(twodads::output_t::o_theta,example.return_pointer_host(), 0.0);

}

