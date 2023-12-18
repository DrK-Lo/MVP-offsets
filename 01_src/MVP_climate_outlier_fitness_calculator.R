#-----------------------------------------------------------------------------------#
# Calculate fitness for all populations to a garden environment with temp optimimum 
#        `opt1` and env2 optimum `opt0`.
#
# Notes
# -----
# - only include environmental optima (eg opt0, opt1) if they are underlying selection
# - validated in 02.04.00
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
# opt1 - temp or MTWetQ optimum for the common garden
# opt0 - (optional) Env2 or MAT optimum for the common garden
# opt2 - MTDQ
# opt3 - PDM
# opt4 - PwarmQ
# opt5 - PWM
# 
#-----------------------------------------------------------------------------------#
library('mvtnorm')
library(progress)
len = length

print(sessionInfo())

args = commandArgs(trailingOnly=TRUE)

seed = as.character(args[1])
output_file = args[2]
opt1 = as.numeric(args[3])  # temp or mat

if (len(args) >= 4){
    opt0 = as.numeric(args[4])  # sal or MTWetQ
}

if (len(args) <= 4){  # MVP sims outlier scenarios
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
} else {  # multivariate sims
    
    opt2 = as.numeric(args[5])  # MTDQ
    opt3 = as.numeric(args[6])  # PDM
    opt4 = as.numeric(args[7])  # PwarmQ
    opt5 = as.numeric(args[8])  # PWM
    
    # find phenos from complex sims (Lotterhos 2023 PNAS)
        # Individuals.txt modified in 02.0500_create_files_for_GF_RDA_RONA_then_train_GF.ipynb -> My_Individuals.txt
    phenodata = read.table('/home/b.lind/offsets/run_20220919_tutorial/tutorial/My_Individuals.txt',
                           header=T)
    traits = c("phenotype1_mat",
               "phenotype2_MTWetQ",
               "phenotype3_MTDQ",
               "phenotype4_PDM",
               "phenotype5_PwarmQ",
               "phenotype6_PWM")
    
#     envdata = read.table(paste0('/home/b.lind/offsets/run_20220919_tutorial/gradient_forests',
#                                 '/training/training_files/tutorial_envfile_GFready_pooled.txt'),
#                          header=T)
    
#     envs = c("env1_mat",
#              "env2_MTWetQ",
#              "env3_MTDQ",
#              "env4_PDM",
#              "env5_PwarmQ",
#              "env6_PWM")
#     opt0 = envdata[1, envs[1]]
#     opt1 = envdata[1, envs[2]]
#     opt2 = envdata[1, envs[3]]
#     opt3 = envdata[1, envs[4]]
#     opt4 = envdata[1, envs[5]]
#     opt5 = envdata[1, envs[6]]
    
    # find opts
}



# create empty dataframe (1 row because input args #2 and #3 specify optima for a single garden ID)
fitness = data.frame(matrix(nrow=1, ncol=100))
colnames(fitness) = 1:ncol(fitness)

# fill in empty dataframe with fitness of each transplant into garden with optima opt0 [opt1] etc
for (transplant_ID in 1:100){
    if (len(args) == 4){  # for two selective environments
        phenos = cbind(subset[subset$subpopID == transplant_ID, 'phen_sal'],
                       subset[subset$subpopID == transplant_ID, 'phen_temp'])
        opts = c(opt0, opt1)
        
        fitness_varcov = matrix(
            c(sigma_k0, 0,
              0, sigma_k1),
            nrow=2
        )
        fitness_norm = dmvnorm(c(0.0, 0.0), c(0.0, 0.0), fitness_varcov)
        
        fits = dmvnorm(
            phenos,
            opts,
            sigma=fitness_varcov
        ) / fitness_norm

    } else if (len(args) == 3) {  # for one selective environment
        phen_temp = subset[subset$subpopID == transplant_ID, 'phen_temp']
        
        fitness_norm = dnorm(0, 0, sigma_k1)
        
        fits = dnorm(phen_temp, opt1, sigma_k1) / fitness_norm

    } else {  # complex sims, six selective environments
#         transplant_ID = 50
        phenos = cbind(phenodata[phenodata$subpopID == transplant_ID, traits[1]],
                       phenodata[phenodata$subpopID == transplant_ID, traits[2]],
                       phenodata[phenodata$subpopID == transplant_ID, traits[3]],
                       phenodata[phenodata$subpopID == transplant_ID, traits[4]],
                       phenodata[phenodata$subpopID == transplant_ID, traits[5]],
                       phenodata[phenodata$subpopID == transplant_ID, traits[6]])
        
        print(c('dim(phenos) = ', dim(phenos)))
        
        print(phenos)
#         print(c('class(phenos[,traits[1]]) = ', class(phenos[,traits[1]])))
        
        sigma_k = 2
        fitness_varcov = matrix(
            c(sigma_k, 0, 0, 0, 0, 0,
              0, sigma_k, 0, 0, 0, 0,
              0, 0, sigma_k, 0, 0, 0,
              0, 0, 0, sigma_k, 0, 0,
              0, 0, 0, 0, sigma_k, 0,
              0, 0, 0, 0, 0, sigma_k
             ),
            nrow=6,
            ncol=6
        )
        
#         print(fitness_varcov)
        
        fitness_norm = dmvnorm(c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                               c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                               fitness_varcov)
        print('fitness_norm')
        print(fitness_norm)
        print(c(opt0, opt1, opt2, opt3, opt4, opt5))
        
        fits = dmvnorm(
            phenos,
            c(opt0, opt1, opt2, opt3, opt4, opt5),
            sigma=fitness_varcov
        ) / fitness_norm
        
    }
    
    
    fits = round(fits, 2)
    
    fitness[1, transplant_ID] = mean(fits)
    
}

write.table(fitness, output_file, sep='	', row.names=T, col.names=T)

cat(sprintf('wrote fitness to %s', output_file))

