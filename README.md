# proPTV_OpenPTV_MyPTV_comparison
 An attempt to merge some knowledge




## Installation instructions

### How to install OpenPTV-Python

if you need to install openptv-python, it's recommended to use conda
Create conda environment with openptv-python

    conda create -n openptv_lineofsight python=3.10
    conda activate openptv_lineofsight

If you run it on the Codespaces, the python is already encapsulated, simply use pip:

    pip install git+https://github.com/openptv/openptv-python.git
    pip install jupyter matplotlib numpy numba


### How to install MyPTV:


    git clone --depth 1 --branch extended_zolof_calibration --single-branch https://github.com/ronshnapp/MyPTV.git
    cd MyPTV
    pip install -e .
    cd ..


### Add interactive matplotlib option 

    pip install ipympl