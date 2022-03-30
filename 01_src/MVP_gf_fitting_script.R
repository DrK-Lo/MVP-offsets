#----------------------------------------------------------------------------------------------------------#
# Given a trained gradient forest, fit model to input climate data, `garden_data`.
#
# Usage
# -----
# Rscript gradient_fitting_script.R gfOut_trainingfile range_file predfile maskfile basename save_dir
#
# Parameters
# ----------
# gfOut_trainingfile - file path to trained gradient forest object, saved as .RDS
# garden_file - tab delimited .txt file path with columns for `lat`, `lon`, and each env used in training
#    - data can be current, future, or uniform (eg for common garden) climate
# predfile - path to RDS of object saved from interpolating gradient forest model across landscape
#    for the same time period as the data that went into training (created in gradient_training.R)
# basename - the basename prefix wanted for saving files in `save_dir`
# save_dir - the directory where files should be saved
#
# Notes
# -----
# - This file is based on gradient_fitting_script.R from github.com/brandonlind/offset_validation
# - Differences include (but may not be limited to):
#     - output not written to .nc file
#     - garden_file input assumed to have row names
#     - garden_file assumes only environmental columns (no extras)
#----------------------------------------------------------------------------------------------------------#

library(gradientForest)


calc_euclid <- function(proj, curr){
    #-------------------------------------------------------------------#
    # CALCULATE EUCLIDEAN DISTANCE BETWEEN CURRENT AND PROJECTED OFFSET
    #-------------------------------------------------------------------#
    
    df <- data.frame(matrix(ncol=ncol(proj), nrow=nrow(proj)))
    stopifnot(all(colnames(proj) == colnames(curr)))
    for (i in 1:ncol(proj)){
        # square each difference between the column data
        df[,i] <- (proj[,i] - curr[,i])**2
    }
    # sum by rows then take the square root of each
    return (sqrt(apply(df, 1, sum)))
}


# COMMAND LINE ARGS
args = commandArgs(trailingOnly=TRUE)
gfOut_trainingfile <- args[1]
garden_file        <- args[2]
predfile           <- args[3]
basename           <- args[4]
save_dir           <- args[5]


# SET UP NAMESPACE
cat(sprintf('\nLoading data ...'))
gfOut <- readRDS(gfOut_trainingfile)  # output from gradient_training.R
garden_data <- read.csv(garden_file, sep='\t', row.names=1)
predOut <- readRDS(predfile)  # output from gradient_training.R


# PREDICT OFFSET FOR COMMON GARDEN
cat(sprintf('\nProjecting to climate of common garden ...'))
projOut <- predict(gfOut, garden_data)


# CALC OFFSET
cat(sprintf('\nCalculating offset ...'))
offset <- calc_euclid(projOut, predOut)
offset_df <- data.frame(offset)
rownames(offset_df) <- rownames(garden_data)


# SAVE PROJECTED OFFSET OBJECT
cat(sprintf('\nSaving offset object ...'))
outfile <- paste0(
    save_dir,
    sprintf('/%s_gradient_forest_fitted.RDS', basename)
)
saveRDS(projOut, outfile)
print(outfile)


# SAVE OFFSET DATA
cat(sprintf('\nSaving offset data ...'))
offsetfile <- paste0(
    save_dir,
    sprintf('/%s_gradient_forest_offset.txt', basename)
)
write.table(offset_df, offsetfile, sep='\t')
print(offsetfile)
