# sim_practice

Code used to develop production code for processessing simulations for the Model Validation Program. These notebooks use seed `1231094`.

---
### Repository Structure

Below are descriptions of the notebooks and scripts in this repo (scripts are used within notebooks). These can be viewed on GitHub but hyperlinks within notebooks won't work. Once we reach production stage, we can link the notebooks to nbviewer.org.

I'll likely develop production scripts from these notebooks, where we take code from these notebooks and put into a script to run on each simulation seed (we can run these scripts within notebooks for record keeping).



__01_train_gradient_forests.ipynb__

- convert simulation files to files neccessary to train Gradient Forests (v0.1-18, [Ellis et al. 2012](https://dx.doi.org/10.1890/11-0252.1) *sensu* [Fitzpatrick & Keller 2015](https://doi.org/10.1111/ele.12376)) using training script from Lind et al. (TODO: LINK).
  -  the files created for pooled data will be used to train RONA (`04_train_RONA.ipynb`).

- create infiles by using individual data (counts of minor allele) or by calculating population-level allele frequencies (ie pooled). For each individual and pooled, create infiles using the whole set up simulated SNPs as well as a set that only includes the loci underlying adaptation.

- sbatch files to server for training

__02_fit_gradient_forests.ipynb__

- after training finishes from `01_train_gradient_forests.ipynb`, fit each trained file to the environment of each of the 100 source populations (ie common gardens)

__03_validate_gradient_forests.ipynb__

- for each individual/pool x adaptive/all datasets, I calculate performance (Spearman's rho<sup>2</sup> between GF offset and simulated fitness) for each common garden (ie across transplanted individuals or pools) as well as for each source population (ie across gardens). I also calculate the slope of the linear regression between `fitness ~ GF_offset`.

- I visualize the performance using a heatmap with coordinates for either garden or source_population coordinates (`x` and `y` from simulation landscape)

- I also visualize the scatter plots for each performance in a similar manner by creating each plot on a map using garden or source_pop coordinates

__04_train_RONA.ipynb__

- for each individual/pool x adaptive/all datasets, I determine which loci have significate linear models with each of the environmental values, and then use these to calculate RONA (RONA is calculate for each pop for each environment).

__05_validate_RONA.ipynb__

- I validate RONA by calculating performance (Spearman's rho<sup>2</sup> between RONA and simulated fitness) within gardens (ie across transplanted pops) and across source populations (ie across gardens).

- I visualize the performance using a heatmap with coordinates for either garden or source_population coordinates (`x` and `y` from simulation landscape)

- I also visualize the scatter plots for each performance in a similar manner by creating each plot on a map using garden or source_pop coordinates

__07_env_importance_from_gradient_forests.ipynb__

- look at rank of environmental importance from the trained Gradient Forest models.







---

`pythonimports` in notebooks can be found here: https://github.com/brandonlind/pythonimports
