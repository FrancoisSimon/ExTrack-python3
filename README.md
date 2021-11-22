ExTrack
-------

This repository contains the necessary scripts to run the method ExTrack. ExTrack is a method to detemine kinetics of particles able to transition between different motion states. It can assess diffusion coefficients, transition rates, localization error as well as annotating the probability for any track to be in each state for every time points.

ExTrack has been designed and implemented by François Simon. The python implementation of ExTrack can profite from GPU parallelization using the cupy library. An additionnal version of ExTrack is available on Fiji via Trackmate thanks to Jean-Yves Tinevez https://sites.imagej.net/TrackMate-ExTrack/. The fiji version can profite from CPU parallelization better performances.

# Dependencies

- numpy
- lmfit
- xmltodict
- matplotlib
- pandas

Optional: jupyter, cupy

For GPU parallelization, cupy can be installed as described here : https://github.com/cupy/cupy. The cupy version will depend on your cuda version which itself must be compatible with your GPU driver and GPU. Usage of cupy requires a change in the extrack module : GPU_computing = False


# Installation (from pip)

`pip install -i https://test.pypi.org/simple/ ExTrack`

# Installation (from this Gitlab) repository

## Install dependencies
`pip install -r requirements.txt`

Alternatively, you can install the dependencies manually by typing:
`pip install numpy lmfit xmltodict`

Optional : `pip install jupyter`

## Install ExTrack

Simply run (as root): `python setup.py install`

Check that it worked: `python -c "import extrack"`

# Tutorial

**Document here how to open a Jupyter notebook**

# Usage
## Main functions

## Extra functions
The `writers` submodule contains some useful functions:

- `mat_to_csv(in_path, out_path)` converts a .mat file to a CSV. It can easily be scripted.

## Input file format
## Caveats

# References

# License
This program is released under the GNU General Public License version 3 or upper (GPLv3+).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Deploying (developer only)
```
python setup.py sdist
gpg --detach-sign -a dist/fastspt-11.4.tar.gz
twine upload dist/fastspt-11.4.tar.gz dist/fastspt-11.4.tar.gz.asc
```
# Authors
Francois Simon

# Bugs/suggestions
Send to bugtracker or to email.
