from setuptools import setup, find_packages

setup(name = 'dysts',
      packages=['dysts'],
      version='0.1',
      # packages=find_packages(),
      install_requires = ["numpy", "scipy", "pandas"],
      extras_require = {
        'full': ['tsfresh', 'nolds', 'numpy', 'sdeint'],
        'data': [
            'dysts_data @ git+https://github.com/williamgilpin/dysts_data'
        ]
      },
      package_dir={'dysts': 'dysts'},
      package_data={'dysts': ['data/*']},
     )

