#-----------------------------------------------------------------------------------------------------------------#
# Run RDA offset analysis sensu Capblancq & Forester 2021.
#
# This script is built to estimate offset to climate outlier scenarios. See MVP_12_RDA_offset.R for function
#     details.
#
# In this script, the `outerdir` trailing arg is not the same `outerdir` used in MVP_12_RDA_offset.R. This 
#     `outerdir` is a completely different directory but assumes some subdirectories from MVP_12_RDA_offset.R's 
#     `outerdir` are symlinked within the `outerdir` arg used here:
#     - used to find gf_training_dir for snp file : paste0(outerdir, '/gradient_forests/training/training_files')
#     - finding muts file : 
#          mutsfile = paste0(paste0(outerdir, '/pca/mutfiles'), sprintf('/%s_pooled_Rout_muts_full.txt', seed))
#     - get pc data : pcfile = paste0(outerdir, 
#                     sprintf("/pca/pca_output/%s_Rout_Gmat_sample_maf-gt-p01_GFready_pooled_all_pca.RDS", seed))
#     - saving offset file : rda_dir = paste0(outerdir, '/rda/offset_outfiles')
#
# Usage
# -----
# conda activate MVP_env_R4.0.3
# Rscript MVP_climate_outlier_RDA_offset.R seed slimdir outerdir rda_file use_RDA_outliers ntraits mvp_dir
#
# Parameters (first 6 args are for MVP_12_RDA_offset.R)
# ----------
# seed
#     - the seed number of the simulation - used to find associated files
# outerdir
#     - the directory under which all files from pipeline are created (--outdir arg to MVP_00.py)
# rda_file
#     - path to an RDS file that contains an RDA object - corrected for structure or not
# use_RDA_outliers
#     - TRUE if script should get outliers from an RDA
#     - FALSE if script should skip outliers and use all loci
#     - CAUSAL if script should use adaptive loci from sims
#     - NEUTRAL if script should use neutral loci from sims
# ntraits
#     - the number of traits (environments) under selection in the simulation seed
# mvp_dir
#     - /path/to/MVP-Offsets/01_src (ie the directory with this script in it) 
#
# Dependencies
# ------------
# MVP_14_climate_outlier_fit_GF.py must be run prior to this script
#
#-----------------------------------------------------------------------------------------------------------------#

# input args
args = commandArgs(trailingOnly=TRUE)

# functions to overwrite MVP_12_RDA_offset.R

# redo sort_dfs_indices
sort_dfs_indices <- function(dfs){
    #------------------------------------------------------------#
    # Sort rownames (subpopID) and columns (future climate val).
    #------------------------------------------------------------#
    rownames(dfs) = sprintf("%05d", as.integer(rownames(dfs)))  # add leading zero for sorting
    dfs = dfs[order(rownames(dfs)), order(as.numeric(colnames(dfs)))]  # sort
    rownames(dfs) = as.character(as.integer(rownames(dfs)))  # remove leading zeros from `subpopID` (ie rownames)
    
    return(dfs)
}


# redo get_garden_climates
get_garden_climates <- function(subsetdf){
    #--------------------------------------------------------------------------------------------------#
    # Get climate outlier values.
    # 
    # Parameters
    # ----------
    # subsetdf - unused (for compatibility with MVP_12_RDA_offset.R :: estimate_garden_offsets)
    # outerdir - (global var) this is the climate_outlier directory (GF scripts created files)
    #
    # Returns
    # -------
    # garden_climates - data.frame, row.names = abs(future climate value), cols = c(temp_opt, sal_opt)
    #
    #--------------------------------------------------------------------------------------------------#
    clim_dir = paste0(dirname(outerdir), '/garden_files')  # dir and files created in MVP_14_climate_outlier_fit_GF.py
    files = list.files(clim_dir, full.names=T)
    
    garden_climates = c()
    for (f in files){
        df = read.table(f, sep='\t', header=T, row.names=1)[1, ]  # get first row (all rows have same vals)
        clim_val_dot_txt = strsplit(basename(f), '_')[[1]][6]
        clim_val = sub('.txt', '', clim_val_dot_txt)
        df$clim_val <- c(clim_val)

        garden_climates = rbind(garden_climates, df[1, ])
    }
    
    return(garden_climates)
}

# redo create_garden_data
create_garden_data <- function(garden_climates, subsetdf, scenario, ntraits){
    #------------------------------------------------------------------------------------------------#
    # For a given climate outlier scenario (`scenario`) create uniform climate for all samples.
    #
    # Parameters
    # ----------
    # garden_climates
    #     - dataframe, pops for rows, climate for columns
    # subsetdf
    #     - dataframe of the subset of individuals chosen from the simulation; one row per individual
    #          columns include climate data (temp_opt and sal_opt)
    # scenario
    #     - ID number for the common garden
    # ntraits
    #     - the number of traits (environments) under selection in the simulation seed
    # ind_or_pooled
    #     - global var; %in% c('ind', 'pooled')
    #------------------------------------------------------------------------------------------------#
    future_clim = garden_climates[scenario, ]
    
    # get a dataframe with rows for pops or inds and cols for envs
    if (ind_or_pooled == 'ind'){
        garden_data = data.frame(temp_opt=rep(future_clim$temp_opt, nrow(subsetdf)),
                                 sal_opt=rep(future_clim$sal_opt, nrow(subsetdf)),
                                 row.names=subsetdf$indID)
    } else {
        garden_data = data.frame(temp_opt=rep(future_clim$temp_opt, len(unique(subsetdf$subpopID))),
                                 sal_opt=rep(future_clim$sal_opt, len(unique(subsetdf$subpopID))),
                                 row.names=unique(subsetdf$subpopID))
    }
    
    # decide how many envs to keep
    garden_data = standardize_envdata(garden_data, ntraits=ntraits)
    
    # make sure it's all one value for each env (col) across rows
    for (env in colnames(garden_data)){
        stopifnot(length(unique(garden_data[, env])) == 1)
    }    
    
    return (garden_data)
}


# redo get_garden_num
get_garden_num <- function(row, garden_climates){
    #------------------------------------------------------------------------------------------#
    # MVP_12_RDA_offset.R :: get_garden_num takes subpopID (`row`) as input and forces integer
    # here future_climate_val could be a float (eg 1.1), so integer would be bad, mkay
    #------------------------------------------------------------------------------------------#
    # temp_opt is always in garden_climates no matter ntraits
    future_climate_val = garden_climates[row, 'temp_opt']
    return(future_climate_val)
}


outlier_main = function(args){
    # input args  (first 6 args are set in MVP_12_RDA_offset.R :: main)
    mvp_dir <- args[7]  # path to MVP_offsets/01_src
    
    # source MVP_12 functions used in RDA
    options(run.main=FALSE)
    mvp_12_rda_offset = paste0(mvp_dir, '/MVP_12_RDA_offset.R')
    source(mvp_12_rda_offset)
    
    # overwrite MVP_12 functions with functions here (eg with get_garden_climates, create_garden_data above)
    thisfile = paste0(mvp_dir, '/MVP_climate_outlier_RDA_offset.R')
    source(thisfile)
    
    # run main from MVP_12_RDA_offset.R
    main(args)
    
}


if (getOption('run.main', default=TRUE)){  # semi-functionally equivalent to python's `if __name__ == '__main__'`
    outlier_main(args)
    
} else {
    remove(args)
}
