#!/usr/bin/python

import dysts
import dysts.flows
from dysts.analysis import *
from dysts.utils import *
from dysts.base import *
# import neurokit2
# from dysts.utils import standardize_ts


# We will make a local copy of the internal database
OUTPUT_FILE = "./chaotic_attractors2.json"
INPUT_FILE = dysts.data_path
# INPUT_FILE = OUTPUT_FILE
RECALCULATE = False
pts_per_trajectory = 3000 # number of timesteps to use to find the jacobian
divide_dt = 5 # factor by which to decrease dt, in order to improve calculation

# double
points_to_sample = 10 # number of initial conditions to sample and average when computing values

with open(INPUT_FILE, "r") as file:
    data = json.load(file)
print(f"Total models being analyzed is {len(data.keys())}.")



for i, item in enumerate(get_attractor_list()):
    print(item, flush=True)
    
    lyap_flag = ("maximum_lyapunov_estimated" not in data[item]) or RECALCULATE
    lyap_flag = True
    print("Find Lyapunov?", lyap_flag)

    corr_flag = ("correlation_dimension" not in data[item]) or RECALCULATE
    corr_flag = True
    print("Find CorrDim?", corr_flag)

    entropy_flag = True
    print("Find MSE?", entropy_flag)
    
    model = getattr(dysts.flows, item)()
    
    ## skip delay models
    if hasattr(model, "delay"):
        if model.delay:
            continue

    print(str(i), item, ": ", end="")
    current_fields = list(data.keys())
    
    sample_pts = sample_initial_conditions(model, points_to_sample)

    all_estimates_lyap = list()
    all_estimates_corrdim = list()
    all_estimates_kydim = list()
    all_estimates_pesin = list()
    all_estimates_mmse = list()
    for j, sample_pt in enumerate(sample_pts):
        print(".", end="")
        model = getattr(dysts.flows, item)()
        model.dt /= divide_dt
        model.ic = sample_pt
        
        if lyap_flag:
            lyapval = find_lyapunov_exponents(model, 5 * pts_per_trajectory, pts_per_period=500)
            all_estimates_lyap.append(lyapval)
            all_estimates_kydim.append(kaplan_yorke_dimension(lyapval))
            all_estimates_pesin.append(np.sum(np.array(lyapval)[np.array(lyapval) > 0]))
            
            
        if corr_flag or entropy_flag:
            sol = model.make_trajectory(3 * pts_per_trajectory, resample=True, pts_per_period=100)
            if corr_flag:
                try:
                    all_estimates_corrdim.append(corr_dim(sol))
                except:
                    print("Bad starting point; ignoring this")
                    pass
            if entropy_flag:
                all_estimates_mmse.append(mse_mv(sol))
                


    if lyap_flag:
        
        # lyap = np.median(np.array(all_estimates_lyap), axis=0)
        # kydim = np.median(all_estimates_kydim)
        # pesin_entropy = np.median(all_estimates_pesin)
        
        ## Weighted average based on quality of estimate
        all_estimates_lyap = np.array(all_estimates_lyap)
        lyap_weights = np.min(np.abs(all_estimates_lyap), axis=1)
        lyap_weights = np.exp(-lyap_weights * 1000)
        lyap_weights /= np.sum(lyap_weights )
        lyap = np.sum(lyap_weights[:, None] * all_estimates_lyap, axis=0) 
        
        kydim = np.sum(lyap_weights * np.array(all_estimates_kydim))
        pesin_entropy = np.sum(lyap_weights * np.array(all_estimates_pesin))
        
        data[item]["maximum_lyapunov_estimated"] = np.max(lyap)
        print(f"lyap: {np.max(lyap)} ", end="")
        
        data[item]["lyapunov_spectrum_estimated"] = list(lyap)
        print(f"spectrum: {lyap} ", end="")
        
        data[item]["kaplan_yorke_dimension"] = kydim
        print(f"kydim: {kydim} ", end="")
        
        data[item]["pesin_entropy"] = pesin_entropy
        print(f"pesin: {pesin_entropy} ", end="")

    if corr_flag:
        if lyap_flag:
            cdim = np.sum(lyap_weights * np.array(all_estimates_corrdim))
        else:
            cdim = np.median(all_estimates_corrdim)
        data[item]["correlation_dimension"] = cdim
        print(f"corr_dim: {cdim} ", end="")
        
    if entropy_flag:
        if lyap_flag:
            ent = np.sum(lyap_weights * np.array(all_estimates_mmse))
        else:
            ent = np.median(all_estimates_mmse)
        data[item]["multiscale_entropy"] = ent
        print(f"multiscale_entropy: {ent} ", end="")


    print("\n", flush=True)


#     break

    # Save new file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=4)

#     except:
#         print(f"Skipped {item}")
#         continue

print("Completed.")

