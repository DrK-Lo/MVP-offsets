
library(LEA)
print(sessionInfo())

args = commandArgs(trailingOnly=TRUE)

rds_file = args[1]

K <- readRDS(rds_file)@K

print(sprintf('K = %s', K))

