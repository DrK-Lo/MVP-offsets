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
# Rscript MVP_RDA_offset.R seed slimdir outerdir rda_file use_RDA_outliers ntraits
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
#     - NEUTRAL if script should use neutral loci from sims
# ntraits
#     - the number of traits (environments) under selection in the simulation seed
#-------------------------------------------------------------------------------------------------------------#

library(data.table)
library(progress)
library(LEA)  # needed to load in previous PCAs
library(vegan)


# options and aliases
options(datatable.fread.datatable=FALSE)
len = length

print(sessionInfo())

# input args
args = commandArgs(trailingOnly=TRUE)


# functions

subset_snps = function(){
    #-----------------------------------------------------------------------------------------------------------#
    # Subset input SNPs to target marker set (all, RDA outliers, or known causal or neutral markers from sims).
    #-----------------------------------------------------------------------------------------------------------#
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
    if (ind_or_pooled == 'ind' | ind_or_pooled == 'pooled' & use_RDA_outliers %in% c('CAUSAL', 'NEUTRAL')){
        # I didn't put mutID or causal_env columns into the pooled muts file
        mutsfile = paste0(slimdir, sprintf('/%s_Rout_muts_full.txt', seed))
    } else {
        mutsfile = paste0(
            paste0(outerdir, '/pca/mutfiles'),
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

    } else if (use_RDA_outliers == 'CAUSAL'){  # get known causal loci from sims
        outliers = muts[muts[, 'mutID'] != 1, 'mutname']
        
    } else {  # get known neutral loci from sims
        outliers = muts[muts[, 'causal_temp'] == 'neutral' & muts[, 'causal_sal'] == 'neutral', 'mutname']
        # at this time, muts did not filter out AF > 0.99 but did filter out AF < 0.01, so I take intersection
            # since I've double checked MAF in MVP_01.py which created the snp file used here
        outliers = intersect(colnames(snps), outliers)
    }

    # subset
    outlier_snps = snps[ , outliers]

    return(outlier_snps)
}


standardize_envdata = function(envdata, ntraits=2){
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
    
    if (ntraits == 1){
        envdata = data.frame(temp=envdata[ , 'temp'],
                             row.names=rownames(envdata))
    }
    
    return(envdata)
}


get_curr_envdata = function(ntraits=2){
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
    env_pres = standardize_envdata(env_pres, ntraits=ntraits)

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
#         pcfile = paste0(
#             slimdir,
#             sprintf("/%s_pca.RDS", seed)
#         )
#         pc_object = readRDS(pcfile)
#         pc_object@projDir = paste0(slimdir, '/')  # it is really dumb the LEA needs to hardcode paths,
#                                                         # just incorporate into one file duh!
        subsetfile = paste0(slimdir, '/', seed, '_Rout_ind_subset.txt')
        subset = read.table(subsetfile)
        stopifnot(all(rownames(env_pres) == subset[ , 'indID']))
        pcdata = subset[ , c('PC1', 'PC2')]

    } else {
        # retrieve PCA - created in MVP-Offsets/01_src/MVP_pooled_pca.R
        pcfile = paste0(
            outerdir,
            sprintf("/pca/pca_output/%s_Rout_Gmat_sample_maf-gt-p01_GFready_pooled_all_pca.RDS", seed)
        )
        pc_object = readRDS(pcfile)
        pcdata = pc_object$projections[ , c(1, 2)]
        colnames(pcdata) = c('PC1', 'PC2')
    }

    rownames(pcdata) <- rownames(env_pres)  # rownames are individual or pop IDs

    return(pcdata)    
}


train_outlier_rda = function(outlier_snps, rda, env_pres, ntraits=2){
    #---------------------------------------------------------------#
    # Train new RDA on outliers inferred from script input args.
    # 
    # Parameters
    # ----------
    # outlier_snps
    #     - output from subset_snps
    # rda
    #     - object loaded from rda_file input arg to this script
    # ntraits
    #     - the number of traits (environments) under selection in the simulation seed
    # 
    # Notes
    # -----
    # - assumes any structure correction is done with first two PCs
    #     - see get_pc_data()
    #---------------------------------------------------------------#
    cat(sprintf('\n\nTraining outlier RDA ...'))

    # decide if I need to estimate a new RDA
    if (use_RDA_outliers == FALSE & ntraits==2){
        # just use the original RDA object - this could be structure-corrected or not
        outlier_rda = rda

    } else {  # estimate a new RDA object with `outlier_snps`

        if (grepl('_RDA_structcorr.RDS', basename(rda_file))){  # this is structure corrected
            # get PCs
            pcdata = get_pc_data(env_pres)

            # replicate Variables object from capblancq & forester
            Variables = data.frame(pcdata, env_pres)

            # get RDA
            if (ntraits == 1){  # use only the adaptive trait (temp)
                outlier_rda = rda(outlier_snps ~ temp + Condition(PC1 + PC2), Variables)
                
            } else {  # use both traits (sal + temp)
                outlier_rda = rda(outlier_snps ~ sal + temp + Condition(PC1 + PC2), Variables)
            }

        } else {  # this is not structure corrected
            # replicate Variables object from capblancq & forester
            Variables = data.frame(env_pres)

            # get RDA
            if (ntraits == 1){  # use only the adaptive trait (temp)
                outlier_rda = rda(outlier_snps ~ temp, Variables)
                
            } else {  # use both traits (sal + temp)
                outlier_rda = rda(outlier_snps ~ sal + temp, Variables)
            }
        }

    }

    return(outlier_rda)
}


ensure_dataframe = function(df, cols){
    #-------------------------------------------------------------------------------------------#
    # Ensure that reducing a data.frame `df` by columns `cols` will return a data.frame object.
    #
    # Notes
    # -----
    # - this was necessary to implement for the function `genomic_offset` when reducing
    #     `var_env_proj_pres` and `var_env_proj_fut` when `ntraits` = 1 (and therefore 
    #      `len(colnames(env_pres))==1` otherwise a single vector was returned when doing eg
    #     df[ , colnames(env_pres)] 
    #-------------------------------------------------------------------------------------------#
    if (len(cols) == 1){
        df = data.frame(df[, cols])
        colnames(df) = cols
    } else if (len(cols) == 0){
        stop("There aren't any cols passed to `ensure_dataframe`")
    } else {
        df = data.frame(df[, cols])
    }
    
    stopifnot(class(df) == 'data.frame')
    
    return(df)
}


genomic_offset = function(RDA, env_pres, env_fut, method="loadings"){
    #------------------------------------------------------------------------#
    # Predict genomic offset using RDA.
    #
    # Notes
    # -----
    # - this was modified from capblancq and forester to allow MVP data
    #     - mostly this means using non-raster transformations or
    #       building in code that can accept either one or two envs (which
    #       cause the RDA to have either one or two axes, respectively)
    #     - additionally, `K` in original function was replaced so that
    #       at most 2 RDA axes were used (ie K=2). In our case, when
    #       we used only one trait (temp) there was only one resulting
    #       RDA axis, and therefore replace `K` with `ncol(env_pres)`.
    #       When using two traits (temp + sal), K=2, and therefore
    #       two RDA axes are used for offset estimation.
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
    
    K = ncol(env_pres)

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
                    ensure_dataframe(var_env_proj_pres, colnames(env_pres)),
                    1,
                    function(x) sum(x * RDA$CCA$biplot[colnames(env_pres) , i])
                )
            )
            Proj_pres[[i]] <- ras_pres
            names(Proj_pres)[i] <- paste0("RDA", as.character(i))

            # Future climates
            ras_fut <- as.vector(
                apply(
        #             var_env_proj_fut[ , names(RDA$CCA$biplot[ , i])],
                    ensure_dataframe(var_env_proj_fut, colnames(env_pres)),
                    1,
        #             function(x) sum(x * RDA$CCA$biplot[ , i])
                    function(x) sum(x * RDA$CCA$biplot[colnames(env_pres) , i])
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
        lapply(1:K, function(x) {Proj_pres[[x]]})
    )

    Proj_offset_pres <- as.data.frame(
        do.call(
            cbind,
            lapply(1:K, function(x) {Proj_offset_pres_[ , x] * weights[x]})
        )
    )

    Proj_offset_fut_ <- do.call(
        cbind,
        lapply(1:K, function(x) {Proj_fut[[x]]})
    )

    Proj_offset_fut <- as.data.frame(
        do.call(
            cbind,
            lapply(1:K, function(x) {Proj_offset_fut_[ , x] * weights[x]})
        )
    )

    # Predict a global genetic offset, incorporating the K first axes weighted by their eigen values
    Proj_offset_global <- unlist(
        lapply(
            1:nrow(Proj_offset_pres),
            function(x) {dist(
                rbind(Proj_offset_pres[x, ],
                      Proj_offset_fut[x, ]),
                method = "euclidean"
            )}
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


get_garden_climates = function(subset, num_expected=100){
    #----------------------------------------------------------------------------------------------------#
    # Get the climate of each common garden by taking mean climate for each individual from that garden.
    #
    # Parameters
    # ----------
    # subset
    #     - dataframe of the subset of individuals chosen from the simulation; one row per individual
    #          columns include climate data (temp_opt and sal_opt)
    # num_expected
    #     - number of expected gardens (N=100)
    #
    # Returns
    # -------
    # garden_climates - data.frame, ncol=n_env, nrow=n_pop
    #----------------------------------------------------------------------------------------------------#
    garden_climates = aggregate(cbind(temp_opt, sal_opt) ~ subpopID, subset, mean)
    rownames(garden_climates) = garden_climates$subpopID
    
    garden_climates = subset(garden_climates, select=-c(subpopID))  # remove subpopID column
    
    stopifnot(nrow(garden_climates) == num_expected)
    
    return(garden_climates)
}


create_garden_data <- function(garden_climates, subset, subpopID, ntraits){
    #------------------------------------------------------------------------------------------------#
    # For a given garden location (`subpopID`) create uniform climate for all samples.
    #
    # Parameters
    # ----------
    # garden_climates
    #     - dataframe, pops for rows, climate for columns
    # subset
    #     - dataframe of the subset of individuals chosen from the simulation; one row per individual
    #          columns include climate data (temp_opt and sal_opt)
    # subpopID
    #     - ID number for the common garden
    # ntraits
    #     - the number of traits (environments) under selection in the simulation seed
    #------------------------------------------------------------------------------------------------#
    stopifnot((1 <= subpopID) & (subpopID <= 100))
    
    # get a dataframe with rows for pops or inds and cols for envs
    if (ind_or_pooled == 'ind'){
        garden_data = data.frame(temp_opt=subset$temp_opt,
                                 sal_opt=subset$sal_opt,
                                 row.names=subset$indID)
    } else {
        garden_data = data.frame(garden_climates)
    }

    # overwrite climate data for all inds or pops with the garden climate
    for (env in c('temp_opt', 'sal_opt')){
        garden_data[, env] = garden_climates[subpopID, env]
    }

    # decide how many envs to keep
    garden_data = standardize_envdata(garden_data, ntraits=ntraits)
    
    # make sure it's all one value for each env (col) across rows
    for (env in colnames(garden_data)){
        stopifnot(length(unique(garden_data[, env])) == 1)
    }
    
    return(garden_data)
}


get_garden_num <- function(subpopID, garden_climates){
    #----------------------------------------------------------------------------------------------------#
    # Add leading zeros to subpopID
    #
    # garden_climates unused - for compatibility with MVP_climate_outlier_RDA_offset.R :: get_garden_num
    #----------------------------------------------------------------------------------------------------#
    garden_num = sprintf("%03d", as.integer(subpopID))
    return(garden_num)
}


sort_dfs_indices <- function(dfs){
    #----------------------------------------------------#
    # Sort rownames (subpopID) and colums (transplatID).
    #----------------------------------------------------#
    rownames(dfs) = sprintf("%05d", as.integer(rownames(dfs)))  # add leading zero for sorting (should have just done as.numeric)
    dfs = dfs[order(rownames(dfs)), order(colnames(dfs))]  # sort
    rownames(dfs) = as.character(as.integer(rownames(dfs)))  # remove leading zeros from `garden_num` (ie rownames)
    colnames(dfs) = as.integer(colnames(dfs))  # remove leading zero
    
    return(dfs)
}


estimate_garden_offsets = function(subset, outlier_rda, env_pres, ntraits=2, num_expected=100){
    #-------------------------------------------------------------------------------------------------#
    # Iterate common garden environments and estimate offset to each environment.
    #
    # Parameters
    # ----------
    # subset
    #     - dataframe of the subset of individuals chosen from the simulation; one row per individual
    #          columns include climate data (temp_opt and sal_opt)
    # outlier_rda
    #     - RDA trained on a specific marker set; output from `train_outlier_rda()`
    # env_pres
    #     - current environment for samples; output from `get_curr_envdata()`
    # ntraits
    #     - the number of traits (environments) under selection in the simulation seed
    # num_expected
    #     - number of expected gardens (N=100)
    #-------------------------------------------------------------------------------------------------#
    
    # get the climate of each common garden
    garden_climates = get_garden_climates(subset)

    # estimate offset to each garden
    prog_bar = progress_bar$new(
        format="estimating offset across gardens [:bar] :current/:total (:percent) eta: :eta",
        total=nrow(garden_climates), clear=FALSE, width=100
    )
    dfs = c()
    offset_lists = list()
    for (subpopID in rownames(garden_climates)){
        # get future climate of common garden
        env_fut = create_garden_data(garden_climates, subset, as.integer(subpopID), ntraits)

        # estimate offset
        offset_list = genomic_offset(outlier_rda, env_pres, env_fut)
        
        # create garden_num from subpopID by adding three leading zeros
        garden_num = get_garden_num(subpopID, garden_climates)

        # separate actual offset prediction
        offset = data.frame(
            offset_list[['Proj_offset_global']]
        )
        rownames(offset) = rownames(env_pres)
        colnames(offset) = c(garden_num)
        
        # combine dfs
        dfs = merge(dfs, offset, by='row.names', all=TRUE)
        rownames(dfs) = dfs[ , 'Row.names']
        dfs = subset(dfs, select=-c(Row.names))
        
        # add offset out to list
        offset_lists[[len(offset_lists) + 1]] = offset_list
        names(offset_lists)[len(offset_lists)] = garden_num

        # update progress bar
        prog_bar$tick(1)
    }
    
    # sort dfs columns and rows
    dfs = sort_dfs_indices(dfs)
    
    # transpose so that source is columns and transplant env is rows
    dfs = t(dfs)
    
    # is this structure-corrected?
    if (grepl('_RDA_structcorr.RDS', basename(rda_file))){
        corr = 'structcorr'
    } else {
        corr = 'nocorr'
    }
    
    # save
    rda_dir = paste0(outerdir, '/rda/offset_outfiles')
    if (!dir.exists(rda_dir)){
        dir.create(rda_dir, recursive=TRUE)
    }
    basename = sprintf("%s_%s_%s_ntraits-%s_%s",
                       seed,
                       ind_or_pooled,
                       use_RDA_outliers,
                       ntraits,
                       corr)
    offset_file = paste0(
        rda_dir, sprintf("/%s_rda_offset.txt", basename)
    )
    write.table(dfs, offset_file, row.names=TRUE, col.names=TRUE)
    cat(sprintf('\n\nwrote offset to: %s', offset_file))
    
    list_file = paste0(
        rda_dir,
        sprintf("/%s_rda_list.RDS", basename)
    )
    saveRDS(offset_lists, list_file)
    cat(sprintf('\n\nwrote rda list to: %s', list_file))
    
}


offset_estimation = function(outlier_snps, rda, ntraits=2){
    #---------------------------------------------------------------#
    # Use `ntraits` to train RDA and estimate offset.
    #
    # Parameters
    # ----------
    # outlier_snps
    #     - subset of all SNPs from sims, output from `subset_snps`
    # rda
    #     - RDA object read in within `main()`
    #---------------------------------------------------------------#
    # get current environmental data (created from MVP_01.py)
    env_pres = get_curr_envdata(ntraits=ntraits)

    # train RDA on outliers
    outlier_rda = train_outlier_rda(outlier_snps, rda, env_pres, ntraits=ntraits)

    # TODO - show adaptive index maps as in capblancq and forester example
    
    # get climate data - rows are individuals
    subset = read.table(paste0(slimdir, '/', seed, '_Rout_ind_subset.txt'))
    stopifnot(length(unique(subset$subpopID)) == 100)  # TODO: remove hard-coded 100 -> num_expected

    # estimate RDA offset for each of the gardens on the simulated landscape
    estimate_garden_offsets(subset, outlier_rda, env_pres, ntraits=ntraits)
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
    ntraits <- as.integer(args[6])  # not set globally

    # directory of files created in MVP_01.py - set globally
    gf_training_dir <<- paste0(outerdir, '/gradient_forests/training/training_files')

    # decide if this is individual or pooled data - set globally
    ind_or_pooled <<- ifelse(grepl('_pooled_', basename(rda_file)), 'pooled', 'ind')

    # read in RDA object trained on all loci (either structure-corrected or not)
    rda = readRDS(rda_file)

    # get the SNPs, subset for outliers (which would be all snps if use_RDA_outliers==FALSE)
    outlier_snps = subset_snps()
    
    # estimate RDA offset for each of the gardens on the simulated landscape
    if (ntraits == 1){
        # estimate RDA using only one env if this env is only env under selection
        cat(sprintf('\n\nEstimating offset for ntraits=1'))
        offset_estimation(outlier_snps, rda, ntraits=1)
    }
    
    # also estimate offset using both envs (one of which could be neutral)
    cat(sprintf('\n\nEstimating offset for ntraits=2'))
    offset_estimation(outlier_snps, rda, ntraits=2)    
    
}


if (getOption('run.main', default=TRUE)){  # semi-functionally equivalent to python's `if __name__ == '__main__'`
    main(args)
} else {
    remove(args)
}
