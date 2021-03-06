#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include "output.h"
//#include "datatype.h"

using namespace std;
using namespace H5;

// Constructor of the base class
output :: output(uint nx_, uint my_) :
    output_counter(0),
    My(my_),
    Nx(nx_)
{
}

output_h5 :: output_h5(uint nx_, uint my_) :
    output(nx_, my_),
    filename("output.h5"),
    output_file(new H5File(H5std_string(filename.data()), H5F_ACC_TRUNC)),
	group_theta(new Group(output_file -> createGroup("/T"))),
    group_theta_x(new Group(output_file -> createGroup("/Tx"))),
    group_theta_y(new Group(output_file -> createGroup("/Ty"))),
	group_omega(new Group(output_file -> createGroup("/O"))),
    group_omega_x(new Group(output_file -> createGroup("/Ox"))),
    group_omega_y(new Group(output_file -> createGroup("/Oy"))),
    group_strmf(new Group(output_file -> createGroup("/S"))),
    group_strmf_x(new Group(output_file -> createGroup("/Sx"))),
    group_strmf_y(new Group(output_file -> createGroup("/Sy"))),
    group_omega_rhs(new Group(output_file -> createGroup("/ORHS"))),
    group_theta_rhs(new Group(output_file -> createGroup("/TRHS"))),
	dspace_file(NULL)
    //o_list(config.get_output())
    // Initialize DataSpace objects in a later for-loop
    // But populate dspace_map with pointers.
{
    o_list.push_back(twodads::output_t::o_theta);
    o_list.push_back(twodads::output_t::o_omega);
    o_list.push_back(twodads::output_t::o_strmf);
	// DataSpace dimension 
	const hsize_t fdim[] = {get_my(), get_nx()};
	// Hyperslab parameter for ghost point array output
	const hsize_t offset[] = {0,0};
	const hsize_t count[] = {get_my(), get_nx()};
	DSetCreatPropList ds_creatplist;  
	dspace_file = new DataSpace(2, count); 

    // populate dspace_map
    dspace_map[twodads::output_t::o_theta] = &dspace_theta;
    //dspace_map[twodads::output_t::o_theta_x] = &dspace_theta_x;
    //dspace_map[twodads::output_t::o_theta_y] = &dspace_theta_y;
    dspace_map[twodads::output_t::o_omega] = &dspace_omega;
    //dspace_map[twodads::output_t::o_omega_x] = &dspace_omega_x;
    //dspace_map[twodads::output_t::o_omega_y] = &dspace_omega_y;
    dspace_map[twodads::output_t::o_strmf] = &dspace_strmf;
    //dspace_map[twodads::output_t::o_strmf_x] = &dspace_strmf_x;
    //dspace_map[twodads::output_t::o_strmf_y] = &dspace_strmf_y;
    //dspace_map[twodads::output_t::o_theta_rhs] = &dspace_theta_rhs;
    //dspace_map[twodads::output_t::o_omega_rhs] = &dspace_omega_rhs;
    // populate field name map

    fname_map[twodads::output_t::o_theta] = "T/";
    //fname_map[twodads::output_t::o_theta_x] = "Tx/";
    //fname_map[twodads::output_t::o_theta_y] = "Ty/";
    fname_map[twodads::output_t::o_omega] = "O/";
    //fname_map[twodads::output_t::o_omega_x] = "Ox/";
    //fname_map[twodads::output_t::o_omega_y] = "Oy/";
    fname_map[twodads::output_t::o_strmf] = "S/";
    //fname_map[twodads::output_t::o_strmf_x] = "Sx/";
    //fname_map[twodads::output_t::o_strmf_y] = "Sy/";
    //fname_map[twodads::output_t::o_theta_rhs] = "TRHS/";
    //fname_map[twodads::output_t::o_omega_rhs] = "ORHS/";
	// Iterate over defined output functions and add them to container
	#ifdef DEBUG
		cout << "Initializing HDF5 output\n";
		cout << "Output file " << output_file -> getFileName() << " created. Id: " << output_file -> getId() << "\n";
	#endif //DEBUG

    DataSpace* dspace;
    for(auto it: o_list)
    {
        dspace = dspace_map[it];
        (*dspace) = DataSpace(2, fdim);
        (*dspace).selectHyperslab(H5S_SELECT_SET, count, offset, NULL, NULL);
    }

    output_file -> flush(H5F_SCOPE_LOCAL);
    output_file -> close();
    delete output_file; 


}



output_h5 :: ~output_h5()
{
    delete dspace_file;
	delete group_theta;
	delete group_theta_x;
	delete group_theta_y;
	delete group_omega;
    delete group_omega_x;
    delete group_omega_y;
	delete group_strmf;
	delete group_strmf_x;
	delete group_strmf_y;
    delete group_theta_rhs;
    delete group_omega_rhs;
}


void output_h5 :: write_output(slab_cuda& slab, twodads::real_t time)
{
    //cuda_array1<cuda::real_t>* arr;
    double* arr;
    // Iterate over list of fields we need to write output for
    for(auto it : o_list)
    {
        // Make sure that get_array_ptr calls copy_device_to_host! 
        slab.copy_device_to_host(it);
        arr = (double*)slab.get_array_ptr(it);
        surface(it, arr, time);
    }
    increment_output_counter();
}



void output_h5 :: surface(twodads::output_t field_name, double* src, const cuda::real_t time)
{
    // update host data on src
    //src -> copy_device_to_host();
    // Dataset name is /[OST]/[0-9]*
    stringstream foo;
    foo << fname_map[field_name] << "/" << to_string(get_output_counter());
    string dataset_name(foo.str());
    output_file = new H5File(filename, H5F_ACC_RDWR);
    DataSpace* dspace_ptr = dspace_map[field_name];

//#ifdef DEBUG
//    cout << "Dataset name: " << dataset_name << "\n";
//#endif //DEBUG
        FloatType float_type(PredType::NATIVE_DOUBLE);
    DataSpace att_space(H5S_SCALAR);
	
	// Create dataset and write data
	DataSet* dataset = new DataSet( 
		output_file->createDataSet(dataset_name, PredType::NATIVE_DOUBLE, *dspace_file, ds_creatplist) );

    // Create time attribute for the Dataset
    Attribute att = dataset -> createAttribute("time", float_type, att_space);
    att.write(float_type, &time);
	dataset -> write( src, PredType::NATIVE_DOUBLE, *dspace_ptr );
	delete dataset;
    delete output_file;
}	


// End of file output.cpp
