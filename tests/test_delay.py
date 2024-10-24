import matplotlib.pyplot as plt

import dysts.flows as dfl


def main():
    dyst_name = "HenonHeiles"
    system = getattr(dfl, dyst_name)()
    sol = system.make_trajectory(
        4096,
        pts_per_period=64,
        resample=True,
        standardize=True,
        embedding_dim=4,
        kind="cubic",
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{system.name} Attractor")
    plt.show()


if __name__ == "__main__":
    main()
