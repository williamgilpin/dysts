.. dysts documentation master file, created by
   sphinx-quickstart on Thu Jul 29 13:59:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##################
dysts API Reference
##################

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Base Classes
===================

.. automodule:: dysts.base
    :members:

Datasets
===================

.. automodule:: dysts.datasets
    :members:
    :exclude-members: load_file, featurize_timeseries

Utilities
===================

.. automodule:: dysts.utils
    :members:
    :exclude-members: group_consecutives, integrate_weiner, parabolic, parabolic_polyfit, signif, resample_timepoints

Analysis
===================

.. automodule:: dysts.analysis
    :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
