.. image:: https://matthieumeo.github.io/hvox/html/_images/hvox.png
  :width: 50 %
  :align: center
  :target: https://matthieumeo.github.io/hvox/html/index

[![License BSD-3](https://img.shields.io/pypi/l/hvox.svg?color=green)](https://github.com/matthieumeo/hvox/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/hvox.svg?color=green)](https://pypi.org/project/hvox)
[![Python Version](https://img.shields.io/pypi/pyversions/hvox.svg?color=green)](https://python.org)

Python reference implementation of the HVOX gridder for mesh-agnostic wide-field interferometry [1].

----------------------------------

## Installation

    # Create conda environment 
    my_env=<CONDA ENVIRONMENT NAME>
    conda create --name "${my_env}"
    
    # Install HVOX
    python -m pip install git+https://github.com/matthieumeo/hvox" 

    # Install additional dependencies to run examples and CLI  
    python -m pip install --index-url=https://artefact.skao.int/repository/pypi-all/simple rascil
    python -m pip install healpy

    # Developer Install HVOX
    python -m pip install "hvox[dev] @ git+https://github.com/matthieumeo/hvox" 

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
`hvox` is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[MIT]: http://opensource.org/licenses/MIT

[file an issue]: https://github.com/matthieumeo/hvox/issues

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## References

[1] [HVOXpaper]Kashani, S., Queralt, J. R., Jarret, A., & Simeoni, M. (2023). HVOX: Scalable Interferometric Synthesis and Analysis of Spherical Sky Maps. arXiv [Cs.CE]. Retrieved from http://arxiv.org/abs/2306.06007