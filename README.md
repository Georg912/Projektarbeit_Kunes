# Projektarbeit by Georg FJ Hufnagl

supervised by Jan Kunes

<!-- TOC -->

-   [Projektarbeit by Georg FJ Hufnagl](#projektarbeit-by-georg-fj-hufnagl)
-   [Getting Started](#getting-started)
    -   [Ubuntu and other Linux Distributions](#ubuntu-and-other-linux-distributions)
    -   [Mac OS](#mac-os)
    -   [Windows](#windows)
-   [Errors and Bugs](#errors-and-bugs)

<!-- /TOC -->

# Getting Started

To use the jupyter notebooks provided in this repository one needs `Python` version 3.7 or higher, but less than 3.11 (Due to `Numba`, a JIT compiler Module to speed up code, currently not available for `Python 3.11`) and also has to download certain Modules (like `Numpy`) to run the code. The required Modules are listed in the `requirements.txt` file.

After finishing the setup for your OS you can start a jupyter notebook server (see [Ubuntu (Linux)](#ubuntu-linux), [Mac OS](#mac-os) or [Windows](#windows) sections below).

The relevant notebooks for playing around are located in the `CCMP/` folder and are named `<number>_<topic_name>.ipynb`. The source code for the algorithms is located in the `Modules` folder.

## Ubuntu (and other Linux Distributions)

Normally, in Ubuntu 20.04 and higher `Python3.8` or newer should be preinstalled. One can check by calling:

    $ python --version
    # or
    $ python3 --version
    # which might return: Python 3.8

If a lower version is installed, call:

    $ sudo apt-get update
    $ sudo apt-get install python3.8 # or higher

And check again the version. If no newer version is available, google how to install higher versions, which should lead to a `deadsnake PPA` or write me an [e-mail](https://github.com/Georg912/Projektarbeit_Kunes/issues) and I can try to help.

To install the required Modules simply run:

    $ pip install -r requirements.txt  --upgrade

within the folder containing `requirements.txt`.

One can now open the `jupyter` notebooks from inside the `CCMP/` folder via

    $ jupyter lab <notebook_name>.ipynb
    # e.g  jupyter lab 01_Dynamical_Systems.ipynb

or

    $ jupyter notebook <notebook_name>.ipynb

I prefer [Jupyterlab](https://jupyterlab.readthedocs.io/en/stable/) as one can customize and extend it compared to simple `Jupyter`.

## Mac OS

I recommend installing python via a package manager like [Homebrew](https://brew.sh/). An installation guide on how to install `Python3` that way can be found [here](https://docs.python-guide.org/starting/install3/osx/).

After installing `Python` and thus `PIP` one can use

    $ pip install -r requirements.txt  --upgrade

to install the required Modules.

One can now open the `jupyter` notebooks (exactly like in linux) via

    $ jupyter lab <notebook_name>.ipynb
    # e.g  jupyter lab 01_Dynamical_Systems.ipynb

and/or

    $ jupyter notebook <notebook_name>.ipynb

I prefer [Jupyterlab](https://jupyterlab.readthedocs.io/en/stable/) as one can customize and extend it compared to simple `Jupyter`.

## Windows

When using Windows I recommend installing `Anaconda3`, because it already has a large number of preinstalled packages and thus one does not have download them separately. The documentation on how to install Anaconda3 can be found [here](https://docs.anaconda.com/anaconda/install/windows/) (I highly recommend adding anaconda to the `PATH` variables in the installation process.)

After finishing the installation one can open an `anaconda prompt` terminal in the folder and simply execute

    $ jupyter lab <notebook_name>.ipynb
    # e.g  jupyter lab 01_Dynamical_Systems.ipynb

and/or

    $ jupyter notebook <notebook_name>.ipynb

Or, if the `PATH` variables have been set, a "normal" `cmd` prompt can also execute the above command to get started

# Errors and Bugs

If you encounter any errors or bugs, please open an issue on the [GitHub repository](https://github.com/Georg912/Projektarbeit_Kunes/issues) or contact me or Prof. Kunes directly via e-mail .
