.. image:: https://matthieumeo.github.io/pycsou/html/_images/hvox.png
  :width: 50 %
  :align: center
  :target: https://matthieumeo.github.io/hvox/html/index

[![License BSD-3](https://img.shields.io/pypi/l/hvox.svg?color=green)](https://github.com/matthieumeo/hvox/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/hvox.svg?color=green)](https://pypi.org/project/hvox)
[![Python Version](https://img.shields.io/pypi/pyversions/hvox.svg?color=green)](https://python.org)

Python reference implementation of the HVOX gridder for mesh-agnostic wide-field interferometry.

----------------------------------

## Installation

    # Install Pycsou-v2-dev (under development)
    my_env=<CONDA ENVIRONMENT NAME>
    
    # Create conda environment and setup dependencies 
    git clone https://github.com/matthieumeo/pycsou && cd pycsou/
    git checkout "v2-dev"
    conda create --name "${my_env}"            \
                 --strict-channel-priority     \
                 --channel=conda-forge         \
                 --file=conda/requirements.txt
    
    # Activate environment and install Pycsou-v2 (under development) 
    conda activate "${my_env}"
    python3 -m pip install .
    
    # Install HVox 
    cd ../
    git clone https://github.com/matthieumeo/hvox && cd hvox/
    python3 -m pip install .


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
