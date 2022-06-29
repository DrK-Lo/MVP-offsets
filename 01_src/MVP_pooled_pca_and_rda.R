#-------------------------------------------------------------------------------------------------#
# Estimate PCs for pooled data for use in RDA_offset.R
#
# Usage
# -----
# conda activate MVP_env_R4.0.3
# Rscript pooled_pca.R seed slimdir snpfile outerdir
#
# Parameters
# ----------
# seed
#     - the seed number of the simulation - used to find associated files
# slimdir
#     - the location of the seed's files output by Katie's post-processing scripts
# snpfile
#     - path to pooled snpfile created for MVP_gf_training_script.R
# outerdir
#     - the directory under which all files from pipeline are created (--outdir arg to MVP_00.py)
#
# Dependencies
# ------------
# - dependent upon completion of MVP_01_train_gradient_forests.py
# - dependend upon completion of MVP_00_start_pipeline.py --rda
# - ability to source MVP_RDA_offset.R for functions:
#     - `get_curr_envdata`
#         - which also calls `standardize_envdata`
#     - `get_pc_data`
#-------------------------------------------------------------------------------------------------#

library(LEA)
library(data.table)
library(vegan)
library(robust)
library(qvalue)

# options and aliases
options(datatable.fread.datatable=FALSE)
len = length

# input args
args = commandArgs(trailingOnly=TRUE)


# functions
prep_snps <- function(snpfile, exclude=c(), sep='\t'){
    #----------------------------------------------------------------#
    # READ IN FREQUENCYxPOP TABLE, REMOVE POPS IN exclude
    #
    # Notes
    # -----
    # - this function is from MVP_gf_training_script.R
    # - assumes snpfile has colname 'index' which has pop names
    # - the rest of the colnames are names for each locus in dataset
    #----------------------------------------------------------------#
    # load data
    snps <- fread(snpfile, sep=sep)
    # set row names
    row.names(snps) <- snps[ , 'index']
    # remove column so all cols are loci
    snps <- subset(snps, select = -c(index))
    # get popnames, exclude irrelevant columns
    pops <- rownames(snps)[!rownames(snps) %in% c('CHROM', 'POS')]
    if (len(exclude) > 0){
        # exclude any other pops
        pops <- pops[!pops %in% exclude]
    }
    # reframe datatable
    snps <- snps[pops, ]
    return(snps)
}

run_pca = function(){
    #--------------------------------------------------#
    # Run principle component analysis on pooled SNPs.
    #--------------------------------------------------#
    
    # create dir if necessary - set globally
    pca_dir <<- paste0(outerdir, '/pca/infiles')
    if (!dir.exists(pca_dir)){
        dir.create(pca_dir, recursive=TRUE)
        print(sprintf('\tcreated pca_dir: %s', pca_dir))
    }

    # read in pooled data
    snps <- prep_snps(snpfile)

    # writing compatable lfmm file for pca input
    tmp_basename = paste0(
        sub('.txt', '', basename(snpfile)),
        '_for_pca.lfmm'
    )
    tmpfile = paste(
        pca_dir,
        tmp_basename,
        sep='/'
    ) 
    write.table(snps, tmpfile, row.names=FALSE, col.names=FALSE, sep='\t')

    # do PCA
    print('Performing PCA ...')
    pc = pca(tmpfile, 30, scale = TRUE)

    # save PCA
    print('Saving PCA ...')
    basename <- sprintf(
        '/%s_pca.RDS',
        sub('.txt', '', basename(snpfile))
    )

    rds_file <- paste0(pca_dir, basename)
    saveRDS(pc, rds_file)
    print('saved pooled PCA to :')
    print(rds_file)
}

rdadapt<-function(rda, K=2){
    #-------------------------------------------------------------------------------#
    # Get RDA SNP outliers.
    #
    # Notes
    # -----
    # - copied from ModelValidationProgram/MVP-NonClinalAF/src/c-AnalyzeSimOutput.R
    # - set default `K` = 2
    #-------------------------------------------------------------------------------#
    zscores = rda$CCA$v[ , 1:as.numeric(K)]
    resscale = apply(zscores, 2, scale)
    resmaha = covRob(resscale, distance=TRUE, na.action=na.omit, estim="pairwiseGK")$dis 
    lambda = median(resmaha) / qchisq(0.5, df=K)
    reschi2test = pchisq(resmaha / lambda, K, lower.tail=FALSE)
    qval = qvalue(reschi2test)
    q.values_rdadapt = qval$qvalues
    
    return(data.frame(p.values=reschi2test, q.values=q.values_rdadapt))
}


create_muts_file = function(rdaout, rdaout_corr){
    #---------------------------------------------------------------------------------------#
    # Create a pooled muts file similar to individual slimdir/{seed}_Rout_muts_full.txt.
    #
    # Notes
    # -----
    # - code modelled after ModelValidationProgram/MVP-NonClinalAF/src/c-AnalyzeSimOutput.R
    #     - Latest commit 1b72723 on May 9, 2022
    # - pooled mutsfile does not contain mutID (ie whether locus was causal in sims)
    #     - handled in MVP_RDA_offset.R
    #
    # Parameters
    # ----------
    # rdaout
    #     - rda object - no structure correction
    # rdaout_corr
    #     - rda object - structure-corrected
    #---------------------------------------------------------------------------------------#
        
    scores = scores(rdaout, choices=1:4)
    scores_corr = scores(rdaout_corr, choices=1:4)

    loci.sc = scores$species
    loci.sc_corr = scores_corr$species
    pool.sc = scores$sites
    pool.sc_corr = scores_corr$sites
    
    muts_full = data.frame(loci.sc[ , 1])
    colnames(muts_full) = c('RDA1_score')
    muts_full$RDA2_score = loci.sc[ , 2]
    muts_full$RDA1_score_corr = loci.sc_corr[ , 1]
    muts_full$RDA2_score_corr = loci.sc_corr[ , 2]
    
    # RDA outliers and error rates####

    ps = rdadapt(rdaout)
    ps_corr = rdadapt(rdaout_corr)

    muts_full$RDA_mlog10P = -log10(ps$p.values)
    muts_full$RDA_mlog10P_sig = ps$q.values<0.05
    muts_full$RDA_mlog10P_corr = -log10(ps_corr$p.values)
    muts_full$RDA_mlog10P_sig_corr = ps_corr$q.values<0.05
    
    muts_full$mutname = rownames(muts_full)
    
    # save pooled muts file
    mutsfile = paste0(
        paste0(outerdir, '/pca'),
        sprintf('/%s_pooled_Rout_muts_full.txt', seed)
    )
    
    write.table(muts_full, mutsfile, row.names=FALSE, col.names=TRUE, sep='\t')
    
    cat(sprintf('\n\nsaved pooled muts file to: %s', mutsfile))
    
}


run_rda = function(){
    #-----------------------------------------------#
    # Run structure-corrected and -uncorrected RDA.
    # 
    # Notes
    # -----
    # - structure correction is done using all loci
    #-----------------------------------------------#
    # read in pooled data
    snps <- prep_snps(snpfile)
    
    # get current environmental data
    env_pres = get_curr_envdata()
    sal = env_pres[, 'sal']
    temp = env_pres[, 'temp']
    
    # read in PCA data
    pcdata = get_pc_data(env_pres)
    
    # structure-uncorrected RDA
    rdaout <- rda(snps ~ sal + temp)
    
    # structure-corrected RDA
    rdaout_corr = rda(snps ~ sal + temp + Condition(pcdata[, 'PC1'], pcdata[, 'PC2']))
    
    # save
    rda_outdir = paste0(outerdir, '/rda/rda_files')
    if (dir.exists(rda_outdir) == FALSE){
        dir.create(rda_outdir, recursive=TRUE)
    }
    file = paste0(
        rda_outdir,
        sprintf('/%s_pooled_RDA.RDS', seed)
    )
    file_corr = paste0(
        rda_outdir,
        sprintf('/%s_pooled_RDA_structcorr.RDS', seed)
    )
    
    saveRDS(rdaout, file)
    saveRDS(rdaout_corr, file_corr)

    # create a muts file like Katie's
    create_muts_file(rdaout, rdaout_corr)
    
}


main = function(args){
    # input args - set globally
    seed <<- args[1]
    slimdir <<- args[2]
    snpfile <<- args[3]
    outerdir <<- args[4]
    mvp_dir <<- args[5]
    ind_or_pooled <<- 'pooled'  # for MVP_RDA_offset::get_curr_envdata
    thisfile = paste0(mvp_dir, '/MVP_pooled_pca_and_rda.R')
    
    # import functions
    options(run.main=FALSE)
    source(paste0(dirname(thisfile), '/MVP_RDA_offset.R'))
    
    # directories of files created in MVP_01.py - set globally for MVP_RDA_offset::get_curr_envdata
    gf_training_dir <<- paste0(outerdir, '/gradient_forests/training/training_files')
    
    # do PCA
    run_pca()
    
    # do RDA
    run_rda()
}


if (getOption('run.main', default=TRUE)){  # semi-functionally equivalent to python's `if __name__ == '__main__'`
    main(args)
    
} else {
    remove(args)
}
