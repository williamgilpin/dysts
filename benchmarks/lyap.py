#!/usr/bin/python

from dysts.base import get_attractor_list
import dysts.flows as dfl
from dysts.analysis import lyapunov_exponent_naive

## open a text file, or make a new one of it doesn't exist
all_lyapunov_exponents = dict()
## load from file if it exists
try:
    with open("lyapunov_exponents.txt", "r") as f:
        for line in f:
            equation_name, lyap = line.split(":")
            all_lyapunov_exponents[equation_name] = float(lyap)
except FileNotFoundError:
    pass

all_attractors = get_attractor_list()
for equation_name in all_attractors:
    ## skip existing entries
    if equation_name in all_lyapunov_exponents.keys():
        print(f"skipping {equation_name}", flush=True)
        continue
    eq = getattr(dfl, equation_name)()
    lyap = lyapunov_exponent_naive(eq)
    print(f"{equation_name}: {lyap}", flush=True)

    all_lyapunov_exponents[equation_name] = lyap

    ## save to file
    with open("lyapunov_exponents.txt", "a") as f:
        f.write(f"{equation_name}: {lyap} \n")
    
    

# from multiprocessing import Pool, Lock

# def compute_lyapunov_exponent_naive(eq, lock):
#     equation_name = eq.name
#     # Check if eq is already in the file
#     with open("lyapunov_exponents.txt", "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             name, value = line.split(": ")
#             if name == eq:
#                 return float(value)

#     # Perform calculation
#     lyap = lyapunov_exponent_naive(eq)

#     # Write result to the file
#     with lock:
#         with open("lyapunov_exponents.txt", "a") as f:
#             f.write(f"{eq}: {lyap}\n")

#     return lyap

# def main():
#     eq_list = [...] # your list of eq objects

#     lock = Lock()

#     with Pool(processes=4) as pool: # use 4 processes
#         results = pool.starmap(lyapunov_exponent_naive, [(eq, lock) for eq in eq_list])
    
#     print(results) # list of lyap values

# if __name__ == '__main__':
#     main()
