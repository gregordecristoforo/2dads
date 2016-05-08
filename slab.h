#ifndef SLAB_H
#define SLAB_H

#include <vector>
#include <iostream>


using namespace std;

namespace twodads{
    enum class output_t {o_theta, o_omega, o_strmf};

    typedef double real_t;
};


namespace cuda{
    typedef double real_t;

};

template <typename T>
class cuda_array1 {
    public:
        cuda_array1(uint, uint);
        T get_value() const;
        const T* get_array_h() const {return data.data();};

    private:
        const uint Nx;
        const uint My;
        vector<double> data;
};


template <typename T> 
cuda_array1<T> :: cuda_array1(uint nx_, uint my_) :
    Nx(nx_), My(my_), data(Nx * My, 7.0)
{ 
}

template <typename T> 
T cuda_array1<T> :: get_value() const
{
    return T(1.0);
};



class slab_cuda {
    public:
        slab_cuda(uint, uint);
        void copy_device_to_host(twodads::output_t);
        cuda_array1<double>* get_array_ptr(twodads::output_t);
    private:
        cuda_array1<double> omega_array;
        cuda_array1<double> theta_array;
};


#endif // SLAB_H
