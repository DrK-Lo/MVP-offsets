#####################################################################################
# Calculate fitness for all populations to a garden environment with temp optimimum 
#        `opt1` and env2 optimum `opt0`.
#
# Notes
# -----
# only include environmental optima (opt0, opt1) if they are underlying selection
#
# Usage
# -----
# conda activate MVP_env_R4.0.3
# Rscript MVP_climate_outlier_fitness_calculator.R seed output_file opt1 [opt2]
# 
# Parameters
# ----------
# seed - simulation seed
# output_file - where to save fitness calculation
# opt1 - temperature optimum for the common garden
# opt2 - (optional) evn2 optimum for the common garden
# 
#####################################################################################
library('mvtnorm')
library(progress)
len = length

print(sessionInfo())

args = commandArgs(trailingOnly=TRUE)

seed = as.character(args[1])
output_file = args[2]
opt1 = as.numeric(args[3])  # temp

if (len(args) == 4){
    opt0 = as.numeric(args[4])  # sal
}

slimdir = '/work/lotterhos/MVP-NonClinalAF/sim_output_20220428/'  # universal directory

# find SIGMA
params_file = '/work/lotterhos/MVP-NonClinalAF/src/0b-final_params-20220428.txt'
params = read.table(params_file, sep=' ', header=T)
rownames(params) = params$seed
sigma_k0 = params[seed, 'SIGMA_K_1']
sigma_k1 = params[seed, 'SIGMA_K_2']

# get phenotype data
subset = read.table(paste0(slimdir, seed, '_Rout_ind_subset.txt'),
                    sep=' ',
                    header=T)

# create empty dataframe (1 row because input args #2 and #3 specify optima for a single garden ID)
fitness = data.frame(matrix(nrow=1, ncol=100))
colnames(fitness) = 1:ncol(fitness)

# fill in empty dataframe with fitness of each transplant into garden with optima opt0 and opt1
for (transplant_ID in 1:100){
    if (len(args) == 4){  # for two selective environments
        phenos = cbind(subset[subset$subpopID == transplant_ID, 'phen_sal'],
                       subset[subset$subpopID == transplant_ID, 'phen_temp'])
        opts = c(opt0, opt1)
        
        fitness_varcov = matrix(c(sigma_k0, 0, 0, sigma_k1), nrow=2)
        fitness_norm = dmvnorm(c(0.0, 0.0), c(0.0, 0.0), fitness_varcov)
        
        fits = dmvnorm(
            phenos,
            opts,
            sigma=fitness_varcov
        ) / fitness_norm
        
    } else {  # for one selective environment
        phen_temp = subset[subset$subpopID == transplant_ID, 'phen_temp']
        
        fitness_norm = dnorm(0, 0, sigma_k1)
        
        fits = dnorm(phen_temp, opt1, sigma_k1) / fitness_norm
    }
    
    fits = round(fits, 2)
    
    fitness[1, transplant_ID] = mean(fits)
    
}

write.table(fitness, output_file, sep='	', row.names=T, col.names=T)


