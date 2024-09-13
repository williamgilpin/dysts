from dysts.flows import MackeyGlass


def main():
    sys = MackeyGlass()
    sol = sys.make_trajectory(1024, pts_per_period=1024 // 10, standardize=True)


if __name__ == "__main__":
    main()
