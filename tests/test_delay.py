import matplotlib.pyplot as plt

from dysts.flows import MackeyGlass


def main():
    sys = MackeyGlass()
    sol = sys.make_trajectory(
        4096,
        pts_per_period=32,
        resample=True,
        standardize=True,
        embedding_dim=2,
        kind="cubic",
    )
    print(sol.shape)

    plt.plot(*sol.T[:2])
    plt.show()


if __name__ == "__main__":
    main()
