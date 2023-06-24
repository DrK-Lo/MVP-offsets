#-------------------------------------------------------------------------------------------------------------------------#
# Use lfmm2 to predict genetic offset to future climates for complex sims.
#
# This script is called from 02_analysis/05_complex_sims/04_train_lfmm.ipynb
#
# Notes
# -----
# similar to MVP_process_lfmm.R but for the complex sims
#
# Usage
# -----
# conda activate MVP_env_R4.0.3
# Rscript MVP_climate_outlier_process_lfmm.R input lfmm_envfile lfmm_envfile_new poplabel_file output_file cand_loci_file
#
# Parameters
# ----------
# input
#     path to the file used for the `input` argument of the lfmm genetic.offset function
# lfmm_envfile
#     path to the file used for the `env` argument of the lfmm genetic.offset function
# lfmm_envfile_new
#     path to the file used for the `new.env` argument of lfmm genetic.offset function
# poplabel_file
#     path to the file to be used for the `pop.labels` argument of the lfmm genetic.offset function
# output_file
#     where to save predicted offset
# cand_loci_file
#     either "None" if using all loci, otherwise path to file to be used for the `candidate.loci` 
#     argument of lfmm genetic.offset function    
#------------------------------------------------------------------------------------------------------------------------#

library(LEA)

print(sessionInfo())

# COMMAND LINE ARGS
args = commandArgs(trailingOnly=TRUE)
# seed             <- args[1]
# slimdir          <- args[2]
# lfmm_envfile     <- args[3]
# lfmm_envfile_new <- args[4]
# poplabel_file    <- args[5]
# output_file      <- args[6]
# cand_loci_file   <- args[7]
input            <- args[1]
lfmm_envfile     <- args[2]
lfmm_envfile_new <- args[3]
poplabel_file    <- args[4]
output_file      <- args[5]
cand_loci_file   <- args[6]



if (cand_loci_file == 'None'){
    cand_loci = NULL
} else {
    cand_loci = as.vector(
        read.csv(cand_loci_file, sep='\t', header=FALSE)[ , "V1"]
    )
}

# basename <- paste0(seed, '_genotypes.lfmm')
# input <- paste(slimdir, basename, sep="/")

print('reading in envfile ...')
env <- read.csv(lfmm_envfile, sep='\t', header=FALSE)

print('reading in transplant envfile ...')
new.env <- read.csv(lfmm_envfile_new, sep='\t', header=FALSE)

print('reading in pop labels ...')
pop.labels <- as.vector(
    read.csv(poplabel_file, sep='\t', header=FALSE)[, "V1"]
)

# print('reading in K ...')
# basename = paste0(seed, '_lfmm2_temp.RDS')  # K will be the same for _lfmm2_temp.RDS or _lfmm2_sal.RDS for this seed
# K <- readRDS(paste(slimdir, basename, sep="/"))@K

print('calculating lfmm2 offset ... ')
offset <- genetic.offset(input=input, env=env, new.env=new.env, pop.labels=pop.labels, K=NULL, candidate.loci=cand_loci)

print('converting offset to data.frame ...')
offset_df = data.frame(offset)

print('saving offset ...')
write.table(offset_df, output_file, col.names=TRUE, row.names=TRUE, sep='\t')

print(output_file)


