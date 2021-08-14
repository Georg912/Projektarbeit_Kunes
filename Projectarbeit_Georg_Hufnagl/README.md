# Projektarbeit by Georg FJ Hufnagl
##  supervised by Jan Kunes

# Getting Started
To use the jupyter notebooks provided in this repository one needs `Python` version 3.6 or higher and also has to download certain Modules (like `Numpy`) to run the code. The required Modules are listed in the `requirements.txt` file.

# Ubuntu (Linux)
Normally, in Ubuntu `Python3.6` should be preinstalled. One can check by calling:

    $ python --version
    # which might return: Python 3.6
If a lower version is installed, call:

    $ sudo apt-get update
    $ sudo apt-get install python3.6
And check again the version. To install the required Modules simply run:

    $ pip install -r requirements.txt  --upgrade
within the folder containing `requirements.txt`.

One can now open the `jupyter` notebooks via

    $ jupyter lab <notebook_name>.ipynb
    # e.g  jupyter lab Dynamical_Systems.ipynb
or

    $ jupyter notebook <notebook_name>.ipynb
I prefer [Jupyterlab](https://jupyterlab.readthedocs.io/en/stable/) as one can customize and extend it compared to simple `Jupyter`.

# Mac OS
I recommend installing python via a package manager like [Homebrew](https://brew.sh/). An installation guide on how to install `PYthon3` that way can be found [here](https://docs.python-guide.org/starting/install3/osx/).

After installing `Python` and thus `PIP` one can use

    $ pip install -r requirements.txt  --upgrade
to install the required Modules.

One can now open the `jupyter` notebooks (exactly like in linus) via

    $ jupyter lab <notebook_name>.ipynb
    # e.g  jupyter lab Dynamical_Systems.ipynb
or

    $ jupyter notebook <notebook_name>.ipynb
I prefer [Jupyterlab](https://jupyterlab.readthedocs.io/en/stable/) as one can customize and extend it compared to simple `Jupyter`.

# Windows