"""
Tests events callables for solve_ivp, using the dysts.base.make_trajectory_ensemble multithreading framework
"""

from typing import Dict, List, Tuple

import numpy as np

from dysts.systems import (
    get_attractor_list,
    make_trajectory_ensemble,
)

DELAY_SYSTEMS = [
    "MackeyGlass",
    "IkedaDelay",
    "SprottDelay",
    "VossDelay",
    "ScrollDelay",
    "PiecewiseCircuit",
]


# Event function to check if integration is taking too long
import time


class TimeLimitEvent:
    def __init__(self, max_duration):
        self.start_time = None
        self.max_duration = max_duration

    def __call__(self, t, y):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration:
            print("Integration stopped due to time limit.")
            return 0  # Trigger the event
        return 1  # Continue the integration


# Event function to detect instability
def instability_event(t, y):
    # Example criterion: If the solution's magnitude exceeds a large threshold
    if np.any(np.abs(y) > 1e6):
        print("y: ", y)
        print("Integration stopped due to instability.")
        return 0  # Trigger the event
    return 1  # Continue the integration


def filter_dict(d: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Filter out keys in a dictionary that have None values,
    corresponding to failed integrations or incomplete trajectory (see DynSys.make_trajectory)
    """
    # List to store the filtered out keys
    excluded_keys = []
    for key in list(d.keys()):
        if d[key] is None:  # or d[key].shape[0] < req_num_vals:
            excluded_keys.append(key)  # Collect the key
            del d[key]  # Remove the key from the dictionary
    print("Keys with insufficent data:", excluded_keys)
    return d, excluded_keys


def main():
    rseed = 999
    num_periods = 5
    num_points = 1024

    # get a list of all available dynamical systems
    systems = get_attractor_list()
    # exclude the delay systems
    systems = [sys for sys in systems if sys not in DELAY_SYSTEMS][:12]

    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=60)  # 1 min time limit
    time_limit_event.terminal = True  # Stop the integration when the event is triggered
    instability_event.terminal = (
        True  # Stop the integration when the event is triggered
    )

    ic_sampler = gaussian_init_cond_sampler(subset=systems, random_seed=rseed)

    # each ensemble is of type Dict[str, [ndarray]]
    dyst_ensemble = make_trajectory_ensemble(
        num_points,
        subset=systems,
        use_multiprocessing=True,
        init_conds=ic_sampler(scale=1e-1),
        param_transform=None,
        use_tqdm=True,
        standardize=True,
        pts_per_period=num_points // num_periods,
        events=[time_limit_event, instability_event],
    )
    dyst_ensemble, excluded_keys = filter_dict(
        dyst_ensemble
    )  # , req_num_vals=num_points)
    if len(excluded_keys) > 0:
        print("INTEGRATION FAILED FOR:", excluded_keys)
    else:
        print("INTEGRATION SUCCEEDED FOR ALL SYSTEMS")


if __name__ == "__main__":
    main()
