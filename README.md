# This code was part of my master thesis and is not used anymore. If you are interested in fluid and plasma simulations I recommend BOUT++ and FELTOR
https://boutproject.github.io/
https://github.com/feltor-dev/feltor

# 2dads
Two-dimensional advection-diffusion solver.

The code is designed to solve advection-diffusion problems with a potential formulation for quasi-two-dimensional flow dynamics
on the GPU.

It uses finite different approximations in one spatial direction and spectral expansion in the other.

Time integration is implemented by stiffly stable schemes with diffusion treated implicitly.

Non-linear advection terms are treated with an energy and enstrophy conserving finite differences scheme.

The code is fully flexible in its treatement of initial and boundary conditions.
