double profile( double* array, double* diag_array, uint Mx, uint num_gp_x, uint Ny, uint num_gp_y){
	int i,j;
	double incr = 0;
	int start = Ny + num_gp_y*num_gp_x/2+num_gp_y/2;
	//double* diag_array;
	//diag_array = (double*)malloc(sizeof(double)*Mx);

	for(i = 0; i<Mx; i++){
	//printf("%d diag is %f\n",i, diag_array[i]);
		for(j=0; j<Ny;j++){
			incr += array[start+i*(Ny+num_gp_y)+j];
			//printf("%d diag is %f\n",j, incr);
		}
	diag_array[i]=incr/(double)Ny;
	incr=0;
	//printf("%d diag is %f\n",i, diag_array[i]);
	}
	return *diag_array;
}

double fluc( double* array, double* profile, double* fluc, uint Mx, uint num_gp_x, uint Ny, uint num_gp_y){
	
	int i,j;
	int start = Ny + num_gp_y*num_gp_x/2+num_gp_y/2;
	//double* fluc;
	//fluc = (double*)malloc(sizeof(double)*Mx);

	for(i = 0; i<Mx; i++){
		for(j=0; j<Ny;j++){
			fluc[start+i*(Ny+num_gp_y)+j] = 
			array[start+i*(Ny+num_gp_y)+j] - profile[i];
		//printf("%d fluc is %f\n",i, fluc[start+i*(Ny+num_gp_y)+j]);
		
	}
	
	//printf("%d fluc is %f\n",i, profile[i]);
	}
	return *fluc;
}

double deriv_y( double* array, double* deriv, uint Mx, uint num_gp_x, uint Ny, uint num_gp_y, double delta_y){


	int i,j;
	int start = Ny + num_gp_y*num_gp_x/2+num_gp_y/2;
	
	for(i = 0; i<Mx; i++){
		
		deriv[start+i*(Ny+num_gp_y)] = 
			(1/(2*delta_y))*(array[start+i*(Ny+num_gp_y)+1] 
			-array[start+i*(Ny+num_gp_y)+Ny-1]);
		

		for(j=1; j<Ny-1;j++){
			deriv[start+i*(Ny+num_gp_y)+j] = 
			(1/(2*delta_y))*(array[start+i*(Ny+num_gp_y)+1+j] 
			-array[start+i*(Ny+num_gp_y)-1+j]);
			
		}
		
		deriv[start+i*(Ny+num_gp_y)+Ny-1] = 
			(1/(2*delta_y))*(array[start+i*(Ny+num_gp_y)] 
			-array[start+i*(Ny+num_gp_y)+Ny-2]);
		

	//printf("%d fluc is %f\n",i, delta_y);
	}
	return *deriv;
}

















