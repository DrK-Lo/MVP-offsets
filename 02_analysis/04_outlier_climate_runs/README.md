Note the *Climate Novelty* workflow is sometimes referred to as "outlier climate" in the code notes and filenames

[Full directory](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/tree/main/02_analysis/04_outlier_climate_runs/)

[00_validate_fitness_calculations.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/00_validate_fitness_calculations.ipynb) - validate script used to calculate fitness of individuals. the script is for calculating fitness in Climate Novelty gardens

[01_calculate_climate_outlier_fitness.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/01_calculate_climate_outlier_fitness.ipynb) - calculate the fitness of populations in the climate outlier scenarios for all 2250 seeds

[02_define_pop_groups.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/02_define_pop_groups.ipynb) - find subpopIDs for populations in northwest corner, range center, and southeast corner (n=9 subpops for each block); this notebook sets up code in MVP_summary_functions to get these pops

[03_submit_climate_jobs.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/03_submit_climate_jobs.ipynb) - just a place to kick off the rest of the climate outlier scenario runs, I did most manually via command line, but wanted to document some examples

[05_visualize_validation_of_outlier_predictions.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/05_visualize_validation_of_outlier_predictions.ipynb) - visualize validation of outlier climate scenarios

[06_multivariate_calculate_climate_outlier_fitness.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/06_multivariate_calculate_climate_outlier_fitness.ipynb) - calculate the fitness of populations in the climate outlier scenarios the multivariate sim

[07_submit_multivariate_jobs.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/07_submit_multivariate_jobs.ipynb) - submit multivariate climate outlier jobs

[08_validate_mutlivariate_lfmm_and_rda.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/08_validate_mutlivariate_lfmm_and_rda.ipynb) - validate climate outlier scenarios for the 6-trait/multivariate sim (kicked off in 02_analysis/04_outlier_climate_runs/07_submit_multivariate_jobs)

[09_visualize_multivariate_outlier_results.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/09_visualize_multivariate_outlier_results.ipynb) - visualize mutivariate climate outlier scenarios

[10_visualize_climate_novelty_PCA.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/04_outlier_climate_runs/10_visualize_climate_novelty_PCA.ipynb) - visualize novelty + within-landscape climates in one PCA
