# MVP Scripts

See docstrings within each script for more info and usage. See functions' docstrings within each script for more detail.

### MVP_00_start_pipeline.py
Script used to kick off the Adaptive Environment workflow.
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

### MVP_04_env_importance_from_gradient_forests.R

From a saved RDS object output from gradient forest training, extract predictor importance and save. Called from 02_analysis/10_supplemental/02_env_importance.ipynb.

### MVP_05_train_RONA.py

Calculate the Risk Of Non-Adaptedness for each subpopulation transplanted to all others. Called from MVP_00_start_pipeline.execute_rona.

### MVP_06_validate_RONA.py

Validate RONA with mean individual fitness per pop from the simulation data. Called from MVP_00_start_pipeline.execute_rona.

### MVP_07_calc_WC_pairwise_FST.py

Calculate population pairwise FST according to Weir & Cockerham 1984.  Called from MVP_00_start_pipeline.execute_fst.

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

Set up scripts to work with MVP_process_lfmm.R scripts created in the MVP offset pipeline, passing new environmental data of climate outlier scenarios. Calls MVP_17_climate_outlier_validate_lfmm.py, MVP_process_lfmm.R OR MVP_complex_sims_process_lfmm.R. Called manually.

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

### MVP_climate_outlier_fitness_calculator.R

Calculate fitness for all populations to a garden environment with temp optimimum

### MVP_climate_outlier_GF_fitting.py

Fit trained GF models to climate outlier climates. Called from MVP_14_climate_outlier_fit_GF.py.

### MVP_climate_outlier_RDA_offset.R

Run RDA offset analysis sensu Capblancq & Forester 2021 on novelty climate scenarios. Called by MVP_18_climate_outlier_rda.py.

### MVP_climate_outlier_RONA_train_and_validate_seed.py

Calculate and validate RONA to climate outlier scenarios for a specific seed. Called by MVP_20_climate_outlier_train_and_validate_RONA.py.

### MVP_complex_sims_process_lfmm.R

Use lfmm2 to predict genetic offset to future climates for complex sims. Called by MVP_16_climate_outlier_lfmm.py.

### MVP_gf_fitting_script.R

Given a trained gradient forest, fit model to input climate data, `garden_data`.

### MVP_gf_training_script.R

Given a set of populations, allele freqs, and environmental data, train gradient forests.

### MVP_nuisance_RDA_offset.R

Run MVP_12_RDA_offset.R but with some functions specific to nuisance experiments. Called from 02_analysis/07_experiments/02_nuisance_envs/06_train_and_validate_nuisance_RDA.ipynb.

### MVP_nuisance_rda_validation.py

Validate offset predictions from RDA (sensu Capblancq & Forester) using population mean fitness. Called from 02_analysis/07_experiments/02_nuisance_envs/06_train_and_validate_nuisance_RDA.ipynb.

### MVP_pooled_pca_and_rda.R

Estimate PCs for pooled data for use in MVP_12_RDA_offset.R. Called from MVP_00_start_pipeline.py.execute_rda.

### MVP_process_lfmm.R

Use lfmm2 to predict genetic offset to future climates. Called from MVP_10_train_lfmm2_offset.py and MVP_16_climate_outlier_lfmm.py.

### MVP_summary_functions.py

API and functions used to create figs across scripts that summarize output. Called from most scripts and notebooks.

### MVP_watch_for_failure_of_train_lfmm2_offset.py

If a job fails, find the commands that failed and try again, repeat process until all jobs are completed successfully. Called from MVP_10_train_lfmm2_offset.py and MVP_16_climate_outlier_lfmm.py.















