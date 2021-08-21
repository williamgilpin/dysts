from setuptools import setup, find_packages

setup(name = 'dysts',
      packages=['dysts'],
      # packages=find_packages(),
      install_requires = ["numpy", "scipy"],
      extras_require = {
        "tsfresh": ["tsfresh"],
        "nolds": ["nolds"],
        "sdeint": ["sdeint"]
      },
      package_dir={'dysts': 'dysts'},
      package_data={'dysts': ['data/*']},
     )