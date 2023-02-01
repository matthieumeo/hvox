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


Developer Install
------------------

It is also possible to install HVOX from the source for developers:


.. code-block:: bash

   >> git clone https://github.com/matthieumeo/hvox
   >> cd <repository_dir>/
   >> pip install -e .

The package documentation can be generated with:

.. code-block:: bash

   >> conda install sphinx=='2.1.*'
   >> python3 setup.py build_sphinx -b singlehtml