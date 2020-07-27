# Installing a Working Version of PYMC3 v3.8 on Windows 10

## OPTIONAL (not needed) - Install mingw-w64
Install `mingw-w64` as administrator, and add path to `g++.exe` to User Environment variables.  For me, the path is `C:\Program Files\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin`.
I am not sure if this step is needed since I also installed the `m2w64-toolchain win-64` using conda as shown below.

## Install Miniconda
1. Download from https://docs.conda.io/en/latest/miniconda.html for your Window 10 configuration (mostly likely 64-bit).  Note, as of 7/27/2020, the latest version of Python is 3.7.
2. Run downloaded file (`Miniconda3-latest-Windows-x86_64.exe`).
   - I installed just for me (user privileges)
   - I chose the default path.
   - I checked the box to <bold>Add Miniconda3 to my PATH environment variable</bold>

## Create New Conda Environment and install M2W64-toolchain
1. Open command prompt and activate the base environment for Miniconda by typing `conda activate base`.
2. Create a new conda environment for PYMC3 with Python 3.7 by typing `conda create -n pm3 python=3.7`.
3. Activate new environment by typing `conda activate pm3`.
4. Test Python version by typing `python` at the command prompt.  Then type `quit`.
5. Install m2w64-toolchain for 64-bit machine by typing `conda install -c conda-forge m2w64-toolchain_win-64`.

## OPTIONAL (not needed) - Install M2W64-toolchain in Base Conda Environment
1. If still in the new environment, type `conda deactivate`.
2. Open command prompt and activate the base environment for Miniconda by typing `conda activate base`.
3. Install m2w64-toolchain for 64-bit machine by typing `conda install -c conda-forge m2w64-toolchain_win-64`.

## Install PYMC3 v3.8 in New `pm3` Conda Environment
1. Activate new environment by typing `conda activate pm3`.
2. Install pymc3 by typing `pip install pymc3==3.8`.
3. Install `libpython` by typing `conda install -c anaconda libpython`.

## Install Jupyter Lab in New `pm3` Conda Environment
1. Ensure new environment `pm3` is active by typing `conda activate pm3`.
2. Install Jupyter Lab by typing `conda install -c conda-forge jupyterlab`.

## Install other packages which pymc3 and arviz might use
1. Install Seaborn by typing `conda install seaborn`.
2. Install Scikit-Learn by typing `conda install scikit-learn`.
3. Install PyJanitor by typing `pip install pyjanitor`.

# BAYES AWAY!!!





