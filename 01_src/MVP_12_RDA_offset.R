#-------------------------------------------------------------------------------------------------------------#
# Run RDA offset analysis sensu Capblancq & Forester 2021.
#
# Notes
# -----
# - code inspired by Caplancq & Forester 2021 Methods in Ecology and Evolution 10.1111/2041-210X.13722 
#    - see also https://github.com/Capblancq/RDA-landscape-genomics/blob/main/RDA_landscape_genomics.html
# - This implementation of RDA offset does not use lat/long in the RDA function.
# - The script MVP_pooled_pca_and_rda.R is dependent upon current implementation of the following functions:
#     - `get_curr_envdata`
#         - which also calls `standardize_envdata`
#     - `get_pc_data`
#
#
# Dependencies
# ------------
# - dependent upon completion of MVP_01_train_gradient_forests.py
# - dependent upon completion of MVP_pooled_pca_and_rda.R
# 
#
# Usage
# -----
# conda activate MVP_env_R4.0.3
# Rscript MVP_RDA_offset.R seed slimdir outerdir rda_file use_RDA_outliers
#
#
# Parameters
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
#
# BIG TODO
# --------
# - figure out pcdata rownames to match envdata, and think about implication to PC object from pooled data
#
# TODO
# ----
# - infer K from data files
# - create figure with RDA outliers
# - create figure as in Capblancq & Forester 2021 to show "adaptive landscape across space"
# - infer PCA from known nuetral markers
#    - for individual data, PCs were estimated using all loci after filtering MAF
#    - for pooled data, PCs were estimated in MVP_pooled_pca_and_rda.R
#-------------------------------------------------------------------------------------------------------------#

library(data.table)
library(progress)
# library(robust)  # needed for rdadapt function
# library(qvalue)  # needed for rdadapt function - installed with BiocManager
library(LEA)  # needed to load in previous PCAs
library(vegan)


# options and aliases
options(datatable.fread.datatable=FALSE)
len = length

print(sessionInfo())

# input args
args = commandArgs(trailingOnly=TRUE)


# functions

subset_snps = function(snp_file){
    #------------------------------------------------------------------------------------------------#
    # Subset input SNPs to target marker set (all, RDA outliers, or known causal markers from sims).
    #------------------------------------------------------------------------------------------------#
    cat(sprintf('\n\nReading in SNPs ...'))
    
    # read in SNPs
    snp_file = paste0(
        gf_training_dir,
        sprintf('/%s_Rout_Gmat_sample_maf-gt-p01_GFready_%s_all.txt',
                seed,
                ind_or_pooled)
    )
    snps = fread(snp_file, sep='\t')
    row.names(snps) = snps[ , 'index']
    snps = subset(snps, select = -c(index))  # remove column so all cols are loci

    # read in simulation data for loci - identifies causal loci and RDA outliers
    if (ind_or_pooled == 'ind' | ind_or_pooled == 'pooled' & use_RDA_outliers == 'CAUSAL'){
        # I didn't put mutID column into the pooled muts file
        mutsfile = paste0(slimdir, sprintf('/%s_Rout_muts_full.txt', seed))
    } else {
        mutsfile = paste0(
            paste0(outerdir, '/pca'),
            sprintf('/%s_pooled_Rout_muts_full.txt', seed)
        )
    }
    muts = read.table(mutsfile, header=TRUE)
    rownames(muts) = muts[ , 'mutname']

    # identify which SNPs to use
    if (use_RDA_outliers == 'TRUE'){  # get RDA outliers according to capblancq & forester 2021
        # determine which column from `muts` to use to ID RDA outliers
        if (grepl('_RDA_structcorr.RDS', basename(rda_file))){  # if structure corrected
            rda_col = 'RDA_mlog10P_sig_corr'
        } else {
            rda_col = 'RDA_mlog10P_sig'
        }
        
        outliers = muts[muts[, rda_col] == TRUE, 'mutname']

    } else if (use_RDA_outliers == 'FALSE'){  # use all SNPs
        outliers = colnames(snps)

    } else {  # get known causal loci from sims
        outliers = muts[muts[, 'mutID'] != 1, 'mutname']
    }

    # subset
    outlier_snps = snps[ , outliers]

    return(outlier_snps)
}


standardize_envdata = function(envdata){
    #-----------------------------------------------------------------------#
    # Standardize environmental names between my implementation and Katie's
    #
    # Notes
    # -----
    # this is necessary because of the names within the RDA objects
    #    - will need to make sure any pooled RDAs takes this into account
    #-----------------------------------------------------------------------#
    cnames = c()
    for (env in colnames(envdata)){
        if (env == 'temp_opt'){
            env = 'temp'
        } else {
            stopifnot(env == 'sal_opt')
            env = 'sal'
        }
        cnames = append(cnames, env)
    }
    colnames(envdata) = cnames
    
    return(envdata)
}


get_curr_envdata = function(){
    #---------------------------------------------------#
    # Get the current environmental data for this seed.
    # 
    # Notes
    # -----
    # - environmental files created in MVP_01.py
    #---------------------------------------------------#
    cat(sprintf('\n\nReading in current environmental data ...'))
    
    # identify envfile and read in
    if (ind_or_pooled == 'ind'){
        basename = sprintf('/%s_envfile_GFready_%s.txt', seed, 'ind')
    } else {
        basename = sprintf('/%s_envfile_GFready_%s.txt', seed, 'pooled')
    }
    envfile = paste0(gf_training_dir, basename)
    env_pres = read.table(envfile, row.names=1, header=TRUE, sep='\t')  # row.names are individual or pop IDs
    
    # format according to env names in sims (I used env_opt, Katie used env - eg temp_opt vs temp)
    env_pres = standardize_envdata(env_pres)

    return(env_pres)
}


get_pc_data = function(env_pres){
    #--------------------------------------------------------#
    # Retrieve PCA loadings for individual data pooled data.
    #
    # Parameters
    # ----------
    # env_pres
    #     - data.frame; rownames are individual or popIDs
    #
    # Notes
    # -----
    # - returns only the first two axes
    #     - assumed by train_outlier_rda
    #--------------------------------------------------------#

    if (ind_or_pooled == 'ind'){
        # retrieve PCA - created from MVP-NonClinalAF/src/c-AnalyzeSimOutput.R
        pcfile = paste0(
            slimdir,
            sprintf("/%s_pca.RDS", seed)
        )

    } else {
        # retrieve PCA - created in MVP-Offsets/01_src/MVP_pooled_pca.R
        pcfile = paste0(
            paste0(outerdir, '/pca/infiles'),
            sprintf("/%s_Rout_Gmat_sample_maf-gt-p01_GFready_pooled_all_pca.RDS", seed)
        )
    }

    pcdata = readRDS(pcfile)$projections[ , c(1, 2)]
    colnames(pcdata) = c('PC1', 'PC2')
    rownames(pcdata) <- rownames(env_pres)  # rownames are individual or pop IDs

    return(pcdata)    
}


train_outlier_rda = function(outlier_snps, rda, env_pres){
    #---------------------------------------------------------------#
    # Train new RDA on outliers inferred from script input args.
    # 
    # Parameters
    # ----------
    # outlier_snps
    #     - output from subset_snps
    # rda
    #     - object loaded from rda_file input arg to this script
    # 
    # Notes
    # -----
    # - assumes any structure correction is done with first two PCs
    #     - see get_pc_data()
    #---------------------------------------------------------------#
    cat(sprintf('\n\nTraining outlier RDA ...'))

    # decide if I need to estimate a new RDA
    if (use_RDA_outliers == FALSE){
        # just use the original RDA object - this could be structure-corrected or not
        outlier_rda = rda
    
    } else {  # estimate a new RDA object with `outlier_snps`
        # get other data
        if (grepl('_RDA_structcorr.RDS', basename(rda_file))){  # this is structure corrected
            # get PCs
            pcdata = get_pc_data(env_pres)
            
            # replicate Variables object from capblancq & forester
            Variables = data.frame(pcdata, env_pres)
            
            # get RDA
            outlier_rda = rda(outlier_snps ~ sal + temp + Condition(PC1 + PC2), Variables)

        } else {  # this is not structure corrected
            # replicate Variables object from capblancq & forester
            Variables = data.frame(env_pres)
            
            # get RDA
            outlier_rda = rda(outlier_snps ~ sal + temp, Variables)
        }

    }

    return(outlier_rda)
}


genomic_offset = function(RDA, env_pres, env_fut, K=2, method="loadings"){
    #------------------------------------------------------------------------#
    # Predict genomic offset using RDA.
    #
    # Notes
    # -----
    # - this was modified from capblancq and forester to allow MVP data
    #     - mostly this means using non-raster transformations
    # - in the original script, 'predict' could be passed to kwarg `method`
    #     - this code was removed because I didn't see it used in their 
    #       example RDA_landscape_genomics.html
    #
    # Parameters
    # ----------
    # RDA
    #     - RDA object
    # env_pres
    #     - data.frame; current environmental values
    #       rows for individuals or pops, columns for envs
    # env_fut
    #     - data.frame; future environmental values
    #       rows for individuals or pops, columns for envs
    # K
    #     - int; number of RDA axes to use in offset calculation
    #     - set to K=2 because of convention used in MVP sims/sim processing
    #------------------------------------------------------------------------#

    # Formatting and scaling environmental rasters for projection
    var_env_proj_pres <- scale(env_pres, center=T, scale=T)
    scale_env = attr(var_env_proj_pres, 'scaled:scale')
    center_env = attr(var_env_proj_pres, 'scaled:center')

    # scale future environment based on scaling of current environment
    var_env_proj_fut = scale(
        env_fut[colnames(env_pres)],
        center_env[colnames(env_pres)],
        scale_env[colnames(env_pres)]            
    )

    # Predicting pixels genetic component based on the loadings of the variables
    if(method == "loadings"){
        # Projection for each RDA axis
        Proj_pres = list()
        Proj_fut = list()
        Proj_offset = list()
        for(i in 1:K){
            # Current climates
            ras_pres = as.vector(
                apply(
                    var_env_proj_pres[ , colnames(env_pres)],
                    1,
                    function(x) sum(x * RDA$CCA$biplot[colnames(env_pres) , i])
                )
            )
            Proj_pres[[i]] <- ras_pres
            names(Proj_pres)[i] <- paste0("RDA", as.character(i))

            # Future climates
            ras_fut <- as.vector(
                apply(
                    var_env_proj_fut[ , names(RDA$CCA$biplot[ , i])],
                    1,
                    function(x) sum(x * RDA$CCA$biplot[ , i])
                )
            )
            Proj_fut[[i]] <- ras_fut
            names(Proj_fut)[i] <- paste0("RDA", as.character(i))
            
            # Single axis genetic offset 
            Proj_offset[[i]] <- abs(Proj_pres[[i]] - Proj_fut[[i]])
            names(Proj_offset)[i] <- paste0("RDA", as.character(i))
        }
    } else {
        stop('method must be "loading"')
    }

    # Weights based on axis eigen values
    weights <- RDA$CCA$eig/sum(RDA$CCA$eig)

    # Weighing the current and future adaptive indices based on the eigen values of the associated axes
    Proj_offset_pres_ <- do.call(
        cbind,
        lapply(1:K, function(x) Proj_pres[[x]])
    )

    Proj_offset_pres <- as.data.frame(
        do.call(
            cbind,
            lapply(1:K, function(x) Proj_offset_pres_[ , x] * weights[x])
        )
    )

    Proj_offset_fut_ <- do.call(
        cbind,
        lapply(1:K, function(x) Proj_fut[[x]])
    )

    Proj_offset_fut <- as.data.frame(
        do.call(
            cbind,
            lapply(1:K, function(x) Proj_offset_fut_[ , x] * weights[x])
        )
    )

    # Predict a global genetic offset, incorporating the K first axes weighted by their eigen values
    Proj_offset_global <- unlist(
        lapply(
            1:nrow(Proj_offset_pres),
            function(x) dist(
                rbind(Proj_offset_pres[x, ],
                      Proj_offset_fut[x, ]),
                method = "euclidean"
            )
        )
    )

    # Return projections for current and future climates for each RDA axis,
        # prediction of genetic offset for each RDA axis and a global genetic offset 
    ret = list(Proj_pres = Proj_pres,
               Proj_fut = Proj_fut,
               Proj_offset = Proj_offset,
               Proj_offset_global = Proj_offset_global,
               weights = weights[1:K])

    return(ret)
}


estimate_garden_offsets = function(outlier_rda, env_pres, num_expected=100){
    #-------------------------------------------------------------------------------#
    # Iterate common garden environments and estimate offset to each environment.
    #
    # Parameters
    # ----------
    # outlier_rda
    #     - RDA trained on a specific marker set; output from `train_outlier_rda()`
    # env_pres
    #     - current environment for samples; output from `get_curr_envdata()`
    # num_expected
    #     - number of expected gardens (N=100)
    #-------------------------------------------------------------------------------#
    cat(sprintf('\n\nEstimating offset'))

    # get common garden environment files
    garden_files = list.files(gf_fitting_dir,
                              pattern=sprintf('_%s_', ind_or_pooled),
                              full.names=TRUE)
    stopifnot(len(garden_files) == num_expected)

    # estimate offset to each garden
    prog_bar = progress_bar$new(
        format = "estimating offset across gardens [:bar] :current/:total (:percent) eta: :eta",
        total = 100, clear = FALSE, width= 100
    )
    for (garden_file in garden_files){
        # which garden (subpopID) is this?
        garden_num_dot_txt = tail(
            strsplit(basename(garden_file), '_')[[1]],
            n=1
        )
        garden_num = sub('.txt', '', garden_num_dot_txt)

        # get future climate of common garden
        env_fut = standardize_envdata(
            read.table(garden_file, header=TRUE)
        )

        # estimate offset
        offset_list = genomic_offset(outlier_rda, env_pres, env_fut)

        # separate actual offset prediction
        offset = data.frame(
            offset_list[['Proj_offset_global']]
        )
        rownames(offset) = rownames(env_pres)
        colnames(offset) = c('Proj_offset_global')

        # save
        rda_dir = paste0(outerdir, '/rda/offset_outfiles')
        if (!dir.exists(rda_dir)){
            dir.create(rda_dir, recursive=TRUE)
        }

        basename = sprintf("%s_%s_%s_%s",
                           seed,
                           garden_num,
                           ind_or_pooled,
                           use_RDA_outliers)
        offset_file = paste0(
            rda_dir, sprintf("/%s_rda_offset.txt", basename)
        )
        write.table(offset, offset_file, row.names=TRUE, col.names=TRUE)

        list_file = paste0(
            rda_dir,
            sprintf("/%s_rda_list.RDS", basename)
        )
        saveRDS(offset_list, list_file)

        # update progress bar
        prog_bar$tick(1)
    }
}


main = function(args){
    #----------------------------------------------------------------------------------------#
    # Function to run full script; allows importing functions into R without running script.
    #----------------------------------------------------------------------------------------#

    # input args - set globally
    seed <<- args[1]
    slimdir <<- args[2]
    outerdir <<- args[3]
    rda_file <<- args[4]
    use_RDA_outliers <<- args[5]

    # directories of files created in MVP_01.py - set globally
    gf_training_dir <<- paste0(outerdir, '/gradient_forests/training/training_files')
    gf_fitting_dir <<- paste0(outerdir, '/gradient_forests/fitting/garden_files')

    # decide if this is individual or pooled data - set globally
    ind_or_pooled <<- ifelse(grepl('_pooled_', basename(rda_file)), 'pooled', 'ind')

    # read in RDA object trained on all loci (either structure-corrected or not)
#     cat(sprintf('\nReading in RDA file ...'))
    rda = readRDS(rda_file)

    # get the SNPs, subset for outliers (which would be all snps if use_RDA_outliers==FALSE)
    outlier_snps = subset_snps()

    # get current environmental data (created from MVP_01.py)
    env_pres = get_curr_envdata()

    # train RDA on outliers
    outlier_rda = train_outlier_rda(outlier_snps, rda, env_pres)

    # TODO - show adaptive index maps as in capblancq and forester example

    # estimate RDA offset for each of the gardens on the simulated landscape
    estimate_garden_offsets(outlier_rda, env_pres)
}


if (getOption('run.main', default=TRUE)){  # semi-functionally equivalent to python's `if __name__ == '__main__'`
    main(args)
} else {
    remove(args)
}
