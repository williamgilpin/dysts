from setuptools import setup

# read the contents of the README file so that PyPI can use it as the long description
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name = 'dysts',
      packages=['dysts'],
      version='0.85',
      # packages=find_packages(),
      install_requires = ["numpy", "scipy", "pandas"],
      extras_require = {
        'full': ['tsfresh', 'nolds', 'numpy', 'sdeint'],
        # 'data': [
        #     'dysts_data @ git+https://github.com/williamgilpin/dysts_data'
        # ]
      },
      package_dir={'dysts': 'dysts'},
      package_data={'dysts': ['data/*']},
      long_description=long_description,
      long_description_content_type='text/markdown'
     )

