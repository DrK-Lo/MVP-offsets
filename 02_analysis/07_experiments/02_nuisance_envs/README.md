[Full directory](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/tree/main/02_analysis/07_experiments/02_nuisance_envs/)

Note that 6-trait datasets are sometimes referred to as "multivariate", "tutorial" or "complex" runs.

[00_create_nuisance_envs_file.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/00_create_nuisance_envs_file.ipynb) - assign nuisance environmental values to each of the 100 subpopIDs

[01_train_GF_seeds_0-225.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/01_train_GF_seeds_0-225.ipynb) - train the first 225 levels of GF with the addition of nuisance envs

[02_fit_and_validate_GF_nuis_envs.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/02_fit_and_validate_GF_nuis_envs.ipynb) - fit trained GF models that used nuisance envs to the common gardens of the environment

[03_train_multivariate_nuis_GF.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/03_train_multivariate_nuis_GF.ipynb) - run nuisance envs on multivariate sims

[04_fit_multivariate_nuis_GF.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/04_fit_multivariate_nuis_GF.ipynb) - fit trained GF models from multivariate sims that used nuisance envs to the common gardens of the environment

[05_visualize_nuisance_GF.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/05_visualize_nuisance_GF.ipynb) - visualize the impact of adding nuisance envs to offset training of Gradient Forests.

[06_train_and_validate_nuisance_RDA.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/06_train_and_validate_nuisance_RDA.ipynb) - train RDA on first 225 seeds for the nuisance experiments

[07_train_nuisance_lfmm.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/07_train_nuisance_lfmm.ipynb) - train lfmm nuisance

[08_visualize_lfmm.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/08_visualize_lfmm.ipynb) - visualize RDA nuisance runs

[09_visualize_rda.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/09_visualize_rda.ipynb) - visualize RDA nuisance runs

[10_validate_GF_mutlivariate_nuisance_gf_compare_with_6trait-6envs.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/10_validate_GF_mutlivariate_nuisance_gf_compare_with_6trait-6envs.ipynb) - validate multivarait sims from GF trained with nuisance envs

[11_env_correlations.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/11_env_correlations.ipynb) - visualize correlation among environmental variables

[12_train_multivariate_lfmm_and_RDA.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/12_train_multivariate_lfmm_and_RDA.ipynb) - train lfmm and RDA on the multivariate sim with nuisance envs

[13_validation_multivariate_nuisance_lfmm.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/13_validation_multivariate_nuisance_lfmm.ipynb) - validate multivariate nuisance LFMM

[14_validation_multivariate_rda.ipynb](https://nbviewer.org/github/ModelValidationProgram/MVP-offsets/blob/main/02_analysis/07_experiments/02_nuisance_envs/14_validation_multivariate_rda.ipynb) - validation multivariate nuisance rda

