.. raw:: html

   <p align="center">
   <img align="center" src="doc/_images/hvox.png" alt="Pyxu logo" width=35%>
   </p>
   <h1> HVOX </h1>

.. image:: https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-blue
   :target: https://www.python.org/downloads/
   :alt: Python 3.9 | 3.10
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT
.. image:: https://img.shields.io/badge/Maturity-Production%2FStable-green.svg
   :target: https://www.python.org/dev/peps/pep-0008/
   :alt: Maturity Level: Production/Stable
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black
.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=flat&logo=pre-commit&logoColor=white
   :target: https://pre-commit.com/
   :alt: pre-commit enabled
.. image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
   :target: https://github.com/matthieumeo/hvox/pulls
   :alt: PRs Welcome



**HVOX** is the Python reference implementation of the HVOX gridder for mesh-agnostic wide-field interferometry [1]

Installation
------------

.. code-block:: bash

   # Create conda environment
   my_env=<CONDA ENVIRONMENT NAME>
   conda create --name "${my_env}"

   # Install HVOX
   python -m pip install git+https://github.com/matthieumeo/hvox

   # Install additional dependencies to run examples and CLI
   python -m pip install --index-url=https://artefact.skao.int/repository/pypi-all/simple rascil
   python -m pip install healpy

   # Developer Install HVOX
   python -m pip install "hvox[dev] @ git+https://github.com/matthieumeo/hvox"


Contributing
------------

Contributions are very welcome. Tests can be run with `tox`_, please ensure the coverage at least stays the same before you submit a pull request.

License
-------

Distributed under the terms of the `MIT`_ license, `hvox` is free and open source software

Issues
------

If you encounter any problems, please `file an issue`_ along with a detailed description.

.. _MIT: http://opensource.org/licenses/MIT
.. _file an issue: https://github.com/matthieumeo/hvox/issues
.. _tox: https://tox.readthedocs.io/en/latest/
.. _pip: https://pypi.org/project/pip/
.. _PyPI: https://pypi.org/

References
----------

.. [1] `HVOXpaper`_ Kashani, S., Queralt, J. R., Jarret, A., & Simeoni, M. (2023). HVOX: Scalable Interferometric Synthesis and Analysis of Spherical Sky Maps. arXiv [Cs.CE]. Retrieved from http://arxiv.org/abs/2306.06007

.. _HVOXpaper: http://arxiv.org/abs/2306.06007
