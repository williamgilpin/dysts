from pathlib import Path

import pkg_resources

data_path = pkg_resources.resource_filename("dysts", "data/chaotic_attractors.json")
data_path2 = pkg_resources.resource_filename("dysts", "data/discrete_maps.json")
data_dirpath = pkg_resources.resource_filename("dysts", "data")
PACKAGEDIR = Path(__file__).parent.absolute()
