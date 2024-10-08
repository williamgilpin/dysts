import dysts.flows as dfl

def main():
    dyst_name = "SprottDelay"
    system = getattr(dfl, dyst_name)()
    sol = system.make_trajectory(
        4096,
        pts_per_period=64,
        resample=True,
        standardize=True,
        embedding_dim=2,
        kind="cubic",
    )
    print(sol)

    print(sol.shape)


if __name__ == "__main__":
    main()
