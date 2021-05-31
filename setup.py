from setuptools import setup, find_packages

setup(name = 'dysts',
      packages=['dysts'],
      # packages=find_packages(),
      package_dir={'dysts': 'dysts'},
      package_data={'dysts': ['data/*']},
      )