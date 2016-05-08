# 2dads
Two-dimensional advection-diffusion solver solver.

The code is designed to for advection-diffusion problems with a potential formulation for quasi-two-dimensional flow dznamics.

The code uses finite different approximations in one spatial direction and spectral expansion in the other.

Time integration is implemented by stiffly stable schemes with diffusion treated implicitly.

Nonlinear advection terms are treated with an energy and enstrophy conserving finite differences scheme.

The code is fully flexible in its treatement of initial and boundary conditions.
