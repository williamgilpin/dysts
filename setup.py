from setuptools import setup, find_packages

setup(name = 'dysts',
      packages=['dysts'],
      version='0.1',
      # packages=find_packages(),
      install_requires = ["numpy", "scipy", "pandas"],
      extras_require = {
        "tsfresh": ["tsfresh"],
        "nolds": ["nolds"],
        "sdeint": ["sdeint"]
      },
      package_dir={'dysts': 'dysts'},
      package_data={'dysts': ['data/*']},
     )
