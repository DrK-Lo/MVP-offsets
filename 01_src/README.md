# MVP Pipeline

Code used to autonomously process output from the MVP Project simulations. Code is assumed to be run in the order presented here; in some cases downstream code depends on files created from upstream execution.

This pipeline assumes all files from a particular seed are in the same folder given as input arguments to .py scripts (i.e., `slimdir`). Scripts then create a directory structure for output as follows:


```
parent_directory (`outdir` arg to MVP_01_train_gradient_forests.py) can handle multiple seed outputs
│
└─── gradient_forests
│   │
│   └─── training
│       │
│       └─── training_files - files used as input to train gradient forests
│       │
│       └─── training_outfiles - RDS R files output from the training script
│       │
│       └─── training_shfiles - sbatch files for, and stdout.out files from, slurm
│   │
│   └─── fitting
│       │
│       └─── fitting_outfiles - offset predictions to each of the subpopulations
│       │
│       └─── garden_files - input files to fitting script - uniform environmental data for each common garden location 
                (rows depend on indSeq or poolSeq)
│       │
│       └─── fitting_shfiles - sbatch files for, and stdout.out files from, slurm
│   │
│   └─── validation
│       │    {seed}_{ind_or_pooled}_{adaptive_or_all}_corrs.pkl - python pickle file that contains all correlations 
                between predicted offset and simulated fitness
│       │   
│       └─── figs - contains heatmaps, histograms, and boxplots that visualize offset performance
│          │ {seed}_*garden_performance* displays relationship between predicted offset for a common garden and fitness 
                of transplants (averaged for individual data)
│          │ {seed}_*garden_slope* 
│          │ {seed}_*source_performance* for samples of a particular subpop, displays relationship between predicted 
                offset to remaining subpops and the fitness in those environments (averaged for individual data)
│
└─── RONA
│   │
│   └─── training
│       │
│       └─── training_files - population-level frequencies of the global minor allele, created in MVP_01.py
│       │
│       └─── training_outfiles 
│         │  {seed}_linear_model_results.pkl - python pickle file of a multidimensional dicitonary that contains the 
                  slope, intercept, and pval from linear models for each locus for each environment
│         │  {seed}_RONA_results.pkl - python pickle file file of a multidimensional dictionary that contains the 
                  estimated RONA for each pop for each env at each transplant garden
│   │
│   └─── validation
│       │
│       └─── figs - heatmaps that show performance within and across sources/common gardens, and slopes of the relationship
│          │ {seed}_garden_performance_heatmap-{env} - displays relationship between predicted offset for a common garden 
                  and fitness of transplants 
│          │ {seed}_garden_slope_heatmap-{env} - displays slope of relationship between predicted offset for a common 
                  garden and fitness of transplants 
│          │ {seed}_source_performance_heatmap-{env} - for samples of a particular subpop, displays relationship between 
                  predicted offset to remaining subpops and the fitness in those environments
│          │ {seed}_source_slope_heatmap-{env} - for samples of a particular subpop, displays slope of relationship between
                  predicted offset to remaining subpops and the fitness in those environments
│       │
│       └─── heatmap_objects - data frames in the form of txt files used to create heatmaps
│
└─── fst
│   │ {seed}_{locus_source}_pairwise_FST.txt - symmetric matrix of population pairwise FST

```

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
  - this environment is used to start the pipeline and run all .py scripts in the repository
  - most python scripts depend on cloning [pythonimports from Brandon Lind](https://github.com/brandonlind/pythonimports). After cloning, create the environment by executing (after updating path):`conda env create -n mvp_env -f /path/to/pythonimports/py385.yml`. The path of the cloned repository will need to be exported to the PYTHONPATH within `$HOME/.bashrc` :

```
export PYTHONPATH="${PYTHONPATH}:/path/to/pythonimports"
```

  - activate the mvp_env environment (`conda activate mvp_env`), then: `conda install -c conda-forge scikit-allel`

**2. Gradient Forests environment**
  - this environment is used to run the GradientForests package v0.1-18
  - create this environment with the following command (updating path): `conda create -n gf_env -f /path/to/MVP-offsets/01_src/gf_env.yml`
  - activate the gf_env environment (`conda activate gf_env`) then install GradientForests: `R CMD INSTALL /path/to/MVP-offsets/01_src/gradientForest_0.1-18.tar.gz`
  - open R, then: 
     -  `install.packages(data.table)`
     -  `install.packages(rgeos)`
     -  `install.packages(raster)`

**3. LFMM2/LEA environment**
  - this environment is used to run lfmm2 from the LEA2 package
  - to retrieve the .yml file clone the [MVP-NonClinalAF repository](https://github.com/ModelValidationProgram/MVP-NonClinalAF/blob/main/src/env/MVP_env_R4.0.3.yml)
  - create this environment with the following command (updating path): `conda create -n lea_env -f /path/to/MVP-NonClinalAF/src/env/MVP_env_R4.0.3.yml`


---
## Scripts
### MVP_01_train_gradient_forests.py

This script creates necessary infiles to train Gradient Forests (GF), and submits slurm jobs to train GF using the script `MVP_gf_training_script.R` for a specific simulation `seed`; and queues up MVP_02.py and MVP_03.py to begin once training is complete. See docstring and code comments for more details.

1. Subset VCF for the samples randomly selected from the full set of sims (~1000 individuals)
2. Create genetic infiles for GF
    - Convert individual genotypes to counts of global minor allele
    - Create a pool-seq-like file that contains population-level frequencies of the global minor allele (to compare to individual input to gradient forests)
    - Using loci known to underlie adaptation in the sims, create a genetic infile that contains only these loci (to compare to output when using all loci in sims)
    - These steps create four files for GF training - 1) indSeq adaptive, 2) indSeq all, 3) poolSeq adaptive, 4) poolSeq all
3. Create environmental infiles for GF for individual and population-level (poolSeq-like) analyses
4. Submits training script for all four training scenarios to slurm. In addition, it creates a second slurm job that is dependent upon the successful completion of all four GF training runs - this dependent job will execute MVP_02.py and MVP_03.py.


### MVP_02_fit_gradient_forests.py

This script should be executed after all four training scenarios from MVP_01 are complete (this is submitted to slurm and will run if training completes successfully for all four training scenarios for a particular seed). It takes the four training scenarios and fits each to every environment in the simulated landscape (as of now, there are 100 populations) using `MVP_gf_fitting_script.R`. This is like predicted offset to the climate of a common garden, where we have common gardens in each subpopulation's environment. See docstring and code comments for more details.

1. Retrieve environmental data from all populations and create files with uniform climate (one for each transplant environment) - these newly created files will be the 'future' climate scenario, one for each subpopulation/common garden location.
2. Fit trained models of GF to these uniform climates

### MVP_03_validate_gradient_forests.py

Using the predicted offset from trained models of GF and the known fitness of individuals (or mean population fitness) within/across subpops to calculate Spearman's rho^2. Visualize performance using figures - see directory structure above. See also docstring and code comments for more details.

### MVP_04_env_importance_from_gradient_forests.py

(TODO - finish translating .ipynb to .py)

Using `MVP_extract_importance.R`, extract environmental importance values inferred from GF training. See docstring and code comments for more details.

### MVP_05_train_RONA.py

(TODO - add in adaptive validation)

Using population-level frequencies of the global minor allele (created in MVP_01.py), estimate the Risk Of Non-Adaptedness (Rellstab et al. 2016). Do so for loci with significant linear models from either the full loci set, or from those known to underlie adaptation in the sims (TODO). See docstring and code comments for more details.

1. For all loci, for each environment, calculate linear models (slope, intercept, pval) between population-level allele frequencies and population climate.
2. Identify loci with significant linear models
3. Use loci with significant linear models to calculate RONA for each pop for each env

### MVP_06_validate_RONA-TODO.py

Using the predicted RONA and population mean fitness within/across subpopulations, calculate Spearman's rho^2 for each env. As with MVP_03.py, visualize within common gardens (transplant fitness vs predicted transplant RONA) and for source populations (how well source pops transplant fitness matched predicted RONA of the remaining gardens). See docstring and code comments for more details.

TODO - do we want to use a Euclidean distance across envs instead of a RONA per env?

### MVP_07_calc_WC_pairwise_FST.py

Using individual-level allele counts for the global minor allele (created in MVP_01.py), calculate population pairwise FST with scikit-allel; population pairwise FST is a prerequisite to estimate General Dissimilarity models (MVP_08.py).


