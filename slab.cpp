#include "slab.h"


slab_cuda :: slab_cuda(uint nx_, uint my_) :
    omega_array(nx_, my_),
    theta_array(nx_, my_)
{
}


void slab_cuda::copy_device_to_host(twodads::output_t)
{
    cout << "slab_cuda::copy_device_to_host(...)" << endl;
}

cuda_array1<double>* slab_cuda::get_array_ptr(twodads::output_t o)
{
    if (o == twodads::output_t::o_theta)
    {
        return &theta_array;
    } else if (o == twodads::output_t::o_omega)
    {
        return &omega_array;
    }
}


