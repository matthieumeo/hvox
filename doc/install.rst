Installation
============

HVOX requires Python 3.6 or greater. It was developed and tested on x86_64 systems running MacOS and Linux.


Dependencies
------------


The package extra dependencies are listed in the files ``requirements.txt``.


Quick Install
-------------

HVOX can be installed very simply via the command:

.. code-block:: bash

   >> pip install git+https://github.com/matthieumeo/hvox.git#egg=hvox

If you have previously activated your conda environment ``pip`` will install HVOX in said environment. Otherwise it will install it in your base environment together with the various dependencies obtained from the file ``requirements.txt``.


Examples ans CLI Install
------------------------

The examples and CLI scripts require additional dependencies that can be installed as:

.. code-block:: bash

   >> git clone https://github.com/matthieumeo/hvox
   >> cd <repository_dir>/
   >> pip install -e .  ".[full]"

Provide access to the command `ms2fits`:

.. code-block:: bash

   >> ms2fits --msname /path/to/input/ms --fitsname /path/to/output/fits


And functionality to the examples `example_dcos` (using direction cosine coordinates) and `example_hpix` (using healpix coordinates).


Developer Install
------------------

It is also possible to install HVOX from the source for developers:

.. code-block:: bash

   >> git clone https://github.com/matthieumeo/hvox
   >> cd <repository_dir>/
   >> pip install -e .  ".[dev]"

The package documentation can be generated with:

.. code-block:: bash

   >> python3 setup.py build_sphinx -b singlehtml

If you need to use HVOX for development project in other projects.