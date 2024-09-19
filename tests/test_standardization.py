import matplotlib.pyplot as plt

import dysts.flows as dfl
from dysts.systems import get_attractor_list


def test_trajectory():
    """visual check for all continuous systems"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for sys_name in get_attractor_list():
        sys = getattr(dfl, sys_name)()
        sol = sys.make_trajectory(1024, pts_per_period=1024 // 10, standardize=True)
        ax.set_title(sys_name)
        ax.plot(*sol.T[:3])
        plt.pause(0.1)
        plt.cla()


def main():
    sys = dfl.Lorenz()
    sol = sys.make_trajectory(1024, pts_per_period=1024 // 10, standardize=True)
    if sol is None:
        raise Exception("Failed to generate trajectory")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*sol.T[:3])  # type: ignore
    plt.show()


if __name__ == "__main__":
    main()
