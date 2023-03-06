# MVP Pipeline

Code used to autonomously process output from the MVP Project simulations. Code is assumed to be run in the order presented here; in some cases downstream code depends on files created from upstream execution, see docstrings.

This pipeline assumes all files from a particular seed are in the same folder given as `slimdir` input arguments given to .py scripts.

---
## Cloning and setting up the MVP-offsets repository for production runs
Before running scripts, users will need to set up the Anaconda environments below. First, however, the MVP-offsets repository will need to be cloned to the local computer.

After cloning, export the path of the 01_src directory to PYTHONPATH within `$HOME/.bashrc`:

```
export PYTHONPATH="${PYTHONPATH}:/path/to/MVP-offsets/01_src"
```

---
## Conda environments
Various [Anaconda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) are used across scripts. Users will need to install Anaconda (not miniconda) and be sure to allow the conda init within $HOME/.bashrc (you should be prompted during install).

**1. Python environment**
  - this python (v3.8.5) environment is used to start the pipeline and run all .py scripts in the repository
  - most python scripts depend on cloning [pythonimports from Brandon Lind](https://github.com/brandonlind/pythonimports). After cloning, create the environment by executing (after updating path):`conda env create -n mvp_env -f /path/to/pythonimports/py385.yml`. The path of the cloned repository will need to be exported to the PYTHONPATH within `$HOME/.bashrc` :

```
export PYTHONPATH="${PYTHONPATH}:/path/to/pythonimports"
```

  - activate the mvp_env environment (`conda activate mvp_env`), then: `conda install -c conda-forge scikit-allel`

**2. Gradient Forests environment**
  - this R (v3.5) environment is used to run the GradientForests package v0.1-18
  - create this environment with the following command (updating path): `conda create -n r35 -f /path/to/MVP-offsets/01_src/gf_env.yml`
  - activate the gf_env environment (`conda activate r35`) then install GradientForests: `R CMD INSTALL /path/to/MVP-offsets/01_src/gradientForest_0.1-18.tar.gz`
  - open R, then: 
     -  `install.packages(data.table)`
     -  `install.packages(rgeos)`
     -  `install.packages(raster)`

**3. LFMM2/LEA environment**
  - this R (v4.0.3) environment is used to run lfmm2 from the LEA2 package
  - to retrieve the .yml file clone the [MVP-NonClinalAF repository](https://github.com/ModelValidationProgram/MVP-NonClinalAF/blob/main/src/env/MVP_env_R4.0.3.yml)
  - create this environment with the following command (updating path): `conda create -n lea_env -f /path/to/MVP-NonClinalAF/src/env/MVP_env_R4.0.3.yml`


---
## Pipelines
### Genetic Offset Pipeline

These scripts (MVP_00.py through MVP_13.py, and their dependencies) use simulations from [Lotterhos 2023](https://doi.org/10.1101/2022.08.03.502621) to train and validate the following genetic offset methods: Gradient Forests, the Risk Of Non-Adaptedness, LFMM, and redundancy analysis-based offset (RDA).

MVP_00_start_pipeline.py is used to kickstart this pipeline. Flags used will determine which offset method is run. See usage below.

### Climate outlier scenario pipeline

These scripts (MVP_14.py through MVP_21.py, and their dependencies) use trained models output from the Genetic Offset Pipeline to predict offset to climate scenarios that do not appear in the training data. Each offset method is run manually; the following will train and validate climate scenarios: MVP_14.py, MVP_16.py, MVP_18.py, MVP_20.py.

---
## Scripts

.

NOTE: See docstrings within each script for more info and usage. See functions' docstrings within each script for more detail.

.

### MVP_00_start_pipeline.py
```
usage: MVP_00_start_pipeline.py -s SLIMDIR -o OUTDIR -e EMAIL -c CONDADIR [--gf] [--rona] [--gdm] [--lfmm] [--rda] [--all] [-h]

optional arguments:
  --gf                  Boolean: true if used, false otherwise.
                        Whether to run Gradient Forests analysis.
  --rona                Boolean: true if used, false otherwise.
                        Whether to run Risk Of Non-Adaptedness analysis.
  --gdm                 Boolean: true if used, false otherwise.
                        Whether to run Generalized Dissimilarity Models.
  --lfmm                Boolean: True if used, False otherwise.
                        Whether to run LFMM2 offset models.
  --rda                 Boolean: True if used, False otherwise.
                        Whether to run RDA offset models.
  --all                 Boolean: True if used, False otherwise.
                        Whether to run all offset analyes.
  -h, --help            Show this help message and exit.

required arguments:
  -s SLIMDIR, --slim-dir SLIMDIR
                        /path/to/directory/with_all/SLiM_simulations
  -o OUTDIR, --outdir OUTDIR
                        /path/to/parent/directory/of_all_output_analyses
                        All analyses will have directories nested under
                        this outdir directory.
  -e EMAIL, --email EMAIL
                        the email address you would like to have slurm
                        notifications sent to
  -c CONDADIR, --condadir CONDADIR
                        /path/to/anaconda3/envs
                        The directory under which all anaconda envs are stored.
```

### MVP_01_train_gradient_forests.py

Train Gradient Forests using simulations from the MVP project. Calls MVP_gf_training_script.R, MVP_02_fit_gradient_forests.py and MVP_03_validate_gradient_forests.py. Called from MVP_00_start_pipeline.execute_gf.


### MVP_02_fit_gradient_forests.py

Fit trained models from gradient forests to the climate of a transplant location (ie the populations in the simulation). Calls MVP_gf_fitting_script.R. Called from MVP_01_train_gradient_forests.py.

### MVP_03_validate_gradient_forests.py

Using the predicted offset from trained models of GF and the known fitness of individuals (or mean population fitness) within/across subpops to calculate Kendall's tau. Visualize performance using figures on a per-seed level. Called from MVP_01_train_gradient_forests.py.

### MVP_05_train_RONA.py

Calculate the Risk Of Non-Adaptedness for each subpopulation transplanted to all others. Called from MVP_00_start_pipeline.execute_rona.

### MVP_06_validate_RONA.py

Validate RONA with mean individual fitness per pop from the simulation data. Called from MVP_00_start_pipeline.execute_rona.

### MVP_10_train_lfmm2_offset.py

Calculate offset using method from lfmm2 for a specific simulation seed. Calls MVP_process_lfmm.R. Called from MVP_00_start_pipeline.execute_lfmm.

### MVP_11_validate_lfmm2_offset.py

Validate offset predictions from lfmm using population mean fitness. Called from MVP_00_start_pipeline.execute_lfmm.

### MVP_12_RDA_offset.R

Run RDA offset analysis sensu Capblancq & Forester 2021. Called from MVP_00.execute_rda. Called from MVP_00_start_pipeline.execute_rda.

### MVP_13_RDA_validation.py

Validate offset predictions from lfmm using population mean fitness. Called from MVP_00.execute_rda.

### MVP_14_climate_outlier_fit_GF.py

Set up climate outlier gradient forest fitting runs using a directory processed through the MVP offset pipeline. Calls MVP_climate_outlier_GF_fitting.py and MVP_15_climate_outlier_validate_GF.py. Called manually.

### MVP_15_climate_outlier_validate_GF.py

Validate climate outlier offset predictions for Gradient Forests. Called from MVP_14_climate_outlier_fit_GF.py.

### MVP_16_climate_outlier_lfmm.py

Set up scripts to work with MVP_process_lfmm.R scripts created in the MVP offset pipeline, passing new environmental data of climate outlier scenarios. Calls MVP_17_climate_outlier_validate_lfmm.py. Called manually.

### MVP_17_climate_outlier_validate_lfmm.py

Validate climate outlier offset predictions for lfmm2. Called from MVP_16_climate_outlier_lfmm.py.

### MVP_18_climate_outlier_rda.py

Set up scripts to work with RDA scripts created in the MVP offset pipeline. Calls MVP_climate_outlier_RDA_offset.R and MVP_19_climate_outlier_validate_RDA.py. Called manually.

### MVP_19_climate_outlier_validate_RDA.py

Validate climate outlier offset predictions for RDA. Called from MVP_18_climate_outlier_rda.py.

### MVP_20_climate_outlier_train_and_validate_RONA.py

Train and validate the Risk Of Non-Adaptedness on outlier climate scenarios. Calls MVP_climate_outlier_RONA_train_and_validate_seed.py. Called manually.

### MVP_21_climate_gather_RONA.py

Gather across seeds from output from MVP_climate_outlier_RONA_train_and_validate_seed.py. Called from MVP_20_climate_outlier_train_and_validate_RONA.py.






















