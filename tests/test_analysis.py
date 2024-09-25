import dysts.flows as dfl
from dysts.analysis import compute_timestep

if __name__ == "__main__":
    dyst_name = "Lorenz"
    system = getattr(dfl, dyst_name)()

    num_periods = 40
    num_points_per_period = 1024
    num_points = num_points_per_period * num_periods

    dt, period = compute_timestep(
        system,
        total_length=num_points,
        transient_fraction=0.1,
        num_iters=5,
        pts_per_period=num_points_per_period,
        timescale="Fourier",
    )
    print("all dt: ", dt)
    print("all periods: ", period)

# TODO: Use scipy optimize for black box optimization of dt from initial guess
# until it meets characteristic timescale criteria
# only do if it is clear how to make a cost function
