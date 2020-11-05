# Installing a Working Version of Facebook Prophet on Windows 10

## Install Miniconda
1. Download from https://docs.conda.io/en/latest/miniconda.html for your Window 10 configuration (mostly likely 64-bit).  Note, as of 7/27/2020, the latest version of Python is 3.7.
2. Run downloaded file (`Miniconda3-latest-Windows-x86_64.exe`).
   - I installed just for me (user privileges)
   - I chose the default path.
   - I checked the box to <bold>Add Miniconda3 to my PATH environment variable</bold>

## Create New Conda Environment and install M2W64-toolchain
1. Open command prompt and activate the base environment for Miniconda by typing `conda activate base`.
2. Create a new conda environment for PyStan with Python 3.7 by typing `conda create -n fbprophet python=3.7`.
3. Activate new environment by typing `conda activate fbprophet`.
4. Test Python version by typing `python` at the command prompt.  Then type `quit`.
5. Install m2w64-toolchain for 64-bit machine by typing `conda install -c conda-forge m2w64-toolchain_win-64`.

## Install Facebook Prophet in New Conda Environment
1. Activate new environment by typing `conda activate fbprophet`.
2. Install PyStan by typing `conda install pystan -c conda-forge`.
3. Install `libpython` by typing `conda install -c anaconda libpython`.

## Install Facebook Prophet
1. Install fbprophet by typing `conda install -c conda-forge fbprophet`.

## Install Jupyter Lab
1. Install jupyter lab by typing `conda install jupyterlab`.




