from setuptools import setup, find_packages

setup(name = 'thom',
      packages=['thom'],
      # packages=find_packages(),
      package_dir={'thom': 'thom'},
      package_data={'thom': ['data/*']},
      )