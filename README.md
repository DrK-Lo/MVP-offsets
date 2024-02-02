# The limits of predicting maladaptation to future environments with genomic data

Anthropogenically driven changes in land use and climate patterns pose unprecedented challenges to species persistence. To understand the extent of these impacts, genomic offset methods have been used to forecast maladaptation of natural populations to future environmental change. However, while their use has become increasingly common, little is known regarding their predictive performance across a wide array of realistic and challenging scenarios. Here, we evaluate four offset methods (Gradient Forests, the Risk-Of-Non-Adaptedness, redundancy analysis, and LFMM2) using an extensive set of simulated datasets that vary demography, adaptive architecture, and the number and spatial patterns of adaptive environments. For each dataset, we train models using either all, adaptive, or neutral marker sets and evaluate performance using in silico common gardens by correlating known fitness with projected offset. Using over 4,850,000 of such evaluations, we find that 1) method performance is largely due to the degree of local adaptation across the metapopulation (*LA*<sub>ΔSA</sub>), 2) adaptive marker sets provide minimal performance advantages, 3) within-landscape performance is variable across gardens and declines when offset models are trained using additional non-adaptive environments, and 4) despite (1), performance declines more rapidly in novel climates for metapopulations with higher *LA*<sub>ΔSA</sub> than lower *LA*<sub>ΔSA</sub>. We discuss the implications of these results for management, assisted gene flow, and assisted migration.

## Usage

If you use or are inspired by code in this repository, please cite the manuscript:
```
Lind BM, KE Lotterhos. 2024. The limits of predicting maladaptation to future environments with genomic data. DOI: https://doi.org/10.1101/2024.01.30.577973
```

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
