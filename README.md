# thom

Simulate hundreds of chaotic dynamical systems.

**If you know of any systems that we should add, please feel free to submit an issue or pull request. Please provide a reference or code that contains all parameter values necessary to produce the claimed dynamics.**


## Implementation details

The default integration step is stored in each continuous-time model's `dt` field. This integration timestep was chosen based on the highest significant frequency observed in the power spectrum, with significance being determined relative to [surrogate time series](https://en.wikipedia.org/wiki/Surrogate_data_testing). The `period` field contains the timescale associated with the dominant frequency in each system's power spectrum.


## To do

+ Currently, I have not found an efficient way to compile the right hand side of each system, ideally whenever the class is first invoked. Rather than using numba or jax, it might be easiest to just implement most of `thom.py` in Cython.
+ Align the initial phases, potentially by picking default starting initial conditions that lie on the attractor, but which are as close as possible to the origin
+ Add a `dimension` field with the number of dynamical equations




