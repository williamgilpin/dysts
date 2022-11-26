import pkg_resources
# from .dysts import *

data_path = pkg_resources.resource_filename('dysts', 'data/chaotic_attractors.json')

data_path2 = pkg_resources.resource_filename('dysts', 'data/discrete_maps.json')

data_dirpath = pkg_resources.resource_filename('dysts', 'data')


from pathlib import Path
PACKAGEDIR = Path(__file__).parent.absolute()



# my_file = pkg_resources.resource_filename('my_data_pack', 'my_data/data_file.txt')

# with open(my_file2) as fin:
#     my_data_object = fin.readlines()