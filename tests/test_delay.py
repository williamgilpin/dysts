import matplotlib.pyplot as plt

from dysts.flows import *


def main():
    sys = SprottDelay()
    sol = sys.make_trajectory(
        4096,
        pts_per_period=64,
        resample=True,
        standardize=True,
        embedding_dim=2,
        kind="cubic",
    )
    print(sol)

    plt.plot(*sol.T[:2])
    plt.show()

    print(sol.shape)


if __name__ == "__main__":
    main()
