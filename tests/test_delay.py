import matplotlib.pyplot as plt

from dysts.flows import MackeyGlass, RoadTraffic


def test_mackey():
    sys = MackeyGlass(tau=20)
    sol = sys.make_trajectory(
        4096,
        pts_per_period=32,
        resample=True,
        standardize=True,
        embedding_dim=2,
        kind="cubic",
    )

    plt.plot(*sol.T)
    plt.show()


def main():
    test_mackey()
    sys = RoadTraffic()
    sol = sys.make_trajectory(
        4096,
    )


if __name__ == "__main__":
    main()
