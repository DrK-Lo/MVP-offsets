# MVP-offsets

Code to process output from MVP Simulations to train and validate various genetic offset methods

## Conda environments
Various [Anaconda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) are used across scripts.  Anaconda (not miniconda) was used for coding environments.

**1. Python environment**
  - this python (v3.8.5) environment is used to run all python scripts (.py) and notebooks (.ipynb) in the repository.
  - most python scripts depend on cloning [pythonimports from Brandon Lind](https://github.com/brandonlind/pythonimports). After cloning, the path of the cloned repository will need to be exported to the PYTHONPATH within `$HOME/.bashrc` :

```
export PYTHONPATH="${PYTHONPATH}:/path/to/pythonimports"
```

  - the following will also need to be added to `$HOME/.bashrc` :
```
export PYTHONPATH="${PYTHONPATH}:/path/to/MVP-offsets/01_src"
```
  - create the mvp_env environment (`conda create -n mvp_env -f /path/to/MVP-offsets/mvp_env.yml`)
  - activate the mvp_env environment (`conda activate mvp_env`), then: `conda install -c conda-forge scikit-allel`

**2. Gradient Forests environment**
  - this R (v3.5) environment is used to run the GradientForests package v0.1-18
  - create this environment with the following command: `conda create -n r35 -f /path/to/MVP-offsets/r35.yml`
  - activate the gf_env environment (`conda activate r35`) then install GradientForests: `R CMD INSTALL /path/to/MVP-offsets/01_src/gradientForest_0.1-18.tar.gz`
  - open R, then: 
     -  `install.packages(data.table)`
     -  `install.packages(rgeos)`
     -  `install.packages(raster)`

**3. LFMM2/LEA + RDA environment**
  - this R (v4.0.3) environment is used to run lfmm2 from the LEA2 package, as well as redundancy analysis (RDA)
  - create this environment with the following command (updating path): `conda create -n MVP_env_R4.0.3 -f /path/to/MVP-offsets/MVP_env_R4.0.3.yml`

---
## Main directories
These two directories hold scripts (01_src) and jupyter notebooks (02_analysis) used for processing and analyzing data.

### 01_src
  - executable scripts, see directory's README for more information

### 02_analysis
  - folders containing jupyter notebooks, see directory's README for more information
